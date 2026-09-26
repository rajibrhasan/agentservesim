"""Replay programs as streaming sessions for engines that act on paused requests.

InferCept's controller decides about a request that is paused for a tool call,
so a program must remain ONE engine request across its turns: turn i+1 arrives
as a StreamingInput appended to the live request, not as a new request. The
chain driver in runner.py submits one request per turn and never creates a
pause, so no InferCept decision can fire under it.

Turn boundaries are observed on the client: turn i is complete when its pinned
output length has streamed back, its tool gap then elapses, and the next turn's
tokens are appended. Timestamps use the event loop clock (time.monotonic()),
the same clock as vLLM's RequestStateStats, so program JCT (arrival -> last
token of the last turn) is comparable with the chain driver's. Engine-internal
per-turn queued/scheduled stamps do not exist for a streaming request and are
recorded as None; each record carries the client-side turn start instead.
"""
import asyncio
import contextlib
import math
import os
from types import SimpleNamespace


def validate_session_inputs(sessions):
    """Require an explicit delta or full-prompt convention.

    We cannot infer deltas by subtracting output lengths: recorded successor
    prompts need not contain the tokens this engine actually generates.
    """
    for session in sessions:
        if session.get('input_mode') not in ('streaming-deltas', 'full-prompts'):
            raise ValueError('InferCept requires input_mode="streaming-deltas": '
                             'the first input is a prompt and subsequent inputs '
                             'are tool-result deltas. Full-prompt replay traces '
                             'must explicitly use input_mode="full-prompts" for reconciliation.')
        for sub in session['sub_requests']:
            ids = sub['input_tok_ids']
            if not ids or int(sub['input_toks']) != len(ids):
                raise ValueError('streaming input_toks must equal its nonempty token list')
            if int(sub['output_toks']) <= 0:
                raise ValueError('streaming output_toks must be positive')


def infercept_engine_kwargs(config, *, num_instances, observations_scheduler):
    """Engine arguments for `--policy-engine-config` with `name: infercept`.

    Every input is an explicit experiment parameter: the waste profile measured
    for this hardware/model/TP, the host-link rate measured on the node, and
    the host pool. Nothing is defaulted from a device table.
    """
    from vllm.config import KVTransferConfig

    missing = [key for key in ('profile', 'bandwidth_bytes_s', 'cpu_bytes_per_rank')
               if key not in config]
    if missing:
        raise ValueError(f'infercept engine config needs {missing}')
    bandwidth = float(config['bandwidth_bytes_s'])
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('bandwidth_bytes_s must be a measured positive rate')
    cpu_bytes = int(config['cpu_bytes_per_rank'])
    scratch = int(config.get('scratch_blocks', 2))
    if cpu_bytes < 0 or scratch < 0 or (cpu_bytes == 0) != (scratch == 0):
        raise ValueError('CPU capacity and staging must both be positive, or both zero for no swap')
    if num_instances != 1:
        raise ValueError('InferCept is a single-engine memory scheduler; use one instance')
    if observations_scheduler:
        raise ValueError('--policy-engine-observations replaces the scheduler and '
                         'cannot combine with the InferCept engine')
    profile = str(config['profile'])
    if not os.path.exists(profile):
        raise FileNotFoundError(f'InferCept waste profile not found: {profile}')
    return {
        'async_scheduling': False,
        'scheduler_cls': 'bench.core.infercept_scheduler.InferceptPolicyScheduler',
        'kv_transfer_config': KVTransferConfig(
            kv_connector='InferceptConnector', kv_role='kv_both',
            kv_connector_module_path='bench.core.infercept_connector',
            kv_connector_extra_config={'cpu_bytes_per_rank': cpu_bytes,
                                       'scratch_blocks': scratch}),
        'additional_config': {'infercept_policy': {
            'profile': profile, 'bandwidth_bytes_s': bandwidth}},
    }


async def submit_all_sessions(engines, sessions, SamplingParams, driver=None, log=None):
    """Replay each program as one streaming request; returns (turn, program) records."""
    validate_session_inputs(sessions)
    from vllm.engine.protocol import StreamingInput
    from vllm.sampling_params import RequestOutputKind

    loop = asyncio.get_event_loop()
    t0_loop = loop.time()
    turn_records, prog_records = [], []

    async def run_session(sidx, sess):
        sid = sess.get('session_id', f'prog{sidx}')
        arrival_ns = int(sess['arrival_time_ns'])
        arrival_ref = t0_loop + arrival_ns / 1e9
        delay = arrival_ref - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        task_started = loop.time()
        subs = sess['sub_requests']
        full_prompts = sess['input_mode'] == 'full-prompts'
        n = len(subs)
        if n == 0:
            prog_records.append({'program_id': sid, 'arrival_ns': arrival_ns,
                                 'jct_ns': None, 'num_turns': 0})
            return

        def params(sub):
            n_out = int(sub['output_toks'])
            # Pinned output length per turn; the session tag routes the request
            # through the InferCept lifecycle instead of vLLM's generic path.
            return SamplingParams(min_tokens=n_out, max_tokens=n_out, ignore_eos=True,
                                  temperature=0.0, output_kind=RequestOutputKind.DELTA,
                                  extra_args={'infercept_session': True, 'program_id': sid,
                                              'infercept_full_prompt': full_prompts})

        instance = 0
        ready_started = loop.time()
        if driver is not None:
            instance, _ = await driver.turn_ready(
                sid, 0, loop.time(), prompt_tokens=len(subs[0]['input_tok_ids']))
        ready_finished = loop.time()
        done = [asyncio.Event() for _ in subs]
        started, first = [None] * n, [None] * n

        async def inputs():
            for i, sub in enumerate(subs):
                if i:
                    await done[i - 1].wait()
                    gap_ns = int(subs[i - 1].get('tool_duration_ns', 0))
                    if gap_ns > 0:
                        await asyncio.sleep(gap_ns / 1e9)
                    if driver is not None:
                        await driver.turn_ready(
                            sid, i, loop.time(), prompt_tokens=len(sub['input_tok_ids']))
                started[i] = loop.time()
                yield StreamingInput(prompt={'prompt_token_ids': list(sub['input_tok_ids'])},
                                     sampling_params=params(sub))
            # Ending the input stream finishes the request after its last turn.
            await done[n - 1].wait()

        turn, count, context, last_end = 0, 0, 0, None
        params_started = loop.time()
        initial_params = params(subs[0])
        params_finished = loop.time()
        async for output in engines[instance].generate(inputs(), initial_params, sid):
            new = len(output.outputs[0].token_ids)
            if new == 0 or turn >= n:
                continue
            now = loop.time()
            if count == 0:
                first[turn] = now
            count += new
            sub = subs[turn]
            n_out = int(sub['output_toks'])
            if count < n_out:
                continue
            if count > n_out:
                raise RuntimeError(f'{sid} turn {turn}: {count} output tokens streamed for a '
                                   f'turn pinned to {n_out}')
            if full_prompts:
                context = int(sub['input_toks']) + n_out
            else:
                context += int(sub['input_toks']) + n_out
            last_end = now
            turn_records.append({
                'request_id': f'{sid}-{turn}', 'program_id': sid, 'turn_idx': turn,
                'input_toks': context - n_out, 'output_toks': n_out,
                'input_delta_toks': None if full_prompts else int(sub['input_toks']),
                'input_mode': sess['input_mode'],
                'arrival_time': started[turn], 'queued_ts': None, 'scheduled_ts': None,
                'first_token_ts': first[turn], 'last_token_ts': now,
                'cached_tokens': None, 'admission_holds': 0, 'streaming_session': True,
                'session_timing': {
                    'arrival_ts': arrival_ref, 'task_started_ts': task_started,
                    'driver_ready_started_ts': ready_started,
                    'driver_ready_finished_ts': ready_finished,
                    'initial_params_started_ts': params_started,
                    'initial_params_finished_ts': params_finished,
                    'first_input_ts': started[0],
                },
            })
            if driver is not None:
                await driver.turn_complete(sid, turn, f'{sid}:{turn}', now - started[turn],
                                           context, instance, now, tool_name=sub.get('tool'))
            done[turn].set()
            turn += 1
            count = 0
        if turn != n:
            raise RuntimeError(f'{sid}: the engine ended the session after {turn} of {n} turns')
        prog_records.append({'program_id': sid, 'arrival_ns': arrival_ns,
                             'jct_ns': int((last_end - arrival_ref) * 1e9), 'num_turns': n})

    if log is not None:
        progress = log.progress('Programs', total=len(sessions))
    else:
        progress = contextlib.nullcontext(SimpleNamespace(advance=lambda: None))
    with progress as bar:
        async def tracked(sidx, sess):
            await run_session(sidx, sess)
            bar.advance()
        await asyncio.gather(*(tracked(i, s) for i, s in enumerate(sessions)))
    prog_records.sort(key=lambda r: r['arrival_ns'])
    return turn_records, prog_records

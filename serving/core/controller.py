import re
from .logger import get_logger

class Controller():
    def __init__(self, total_num):
        self.end_dict = {}
        self.total_num = total_num
        self.logger = get_logger(self.__class__)
        for i in range(total_num):
            self.end_dict[i] = -1


    def read_wait(self, p):
        out = [""]
        while "Waiting" not in out[-1] and out[-1] != "Checking Non-Exited Systems ...\n":
            line = p.stdout.readline()
            # For debugging
            # print(line, end='')
            if line == "":  # EOF: backend died; readline would otherwise spin forever
                self._raise_backend_dead(p, out)
            out.append(line)
            p.stdout.flush()
        return out

    def check_end(self, p):
        out = ["",""]
        while out[-2] != "All Request Has Been Exited\n" and out[-2] != "ERROR: Some Requests Remain\n":
            line = p.stdout.readline()
            if line == "":
                self._raise_backend_dead(p, out)
            out.append(line)
            p.stdout.flush()
        print(out[-4], end='')
        print(out[-2], end='')
        return out

    def _raise_backend_dead(self, p, out=None):
        rc = p.poll()
        if rc is None:
            try:
                rc = p.wait(timeout=5)
            except Exception:
                pass
        # ASTRA-Sim is spawned with stderr=PIPE, so whatever it printed on the
        # way down (a C++ assertion, bad_alloc, a signal name) lands there and
        # used to be discarded, leaving a bare exit code to debug from.  We
        # only get here once stdout hit EOF, i.e. the backend is already gone,
        # so draining the pipe returns what is buffered and then EOF.
        tail = ""
        try:
            if p.stderr is not None:
                lines = [l for l in (p.stderr.read() or "").splitlines() if l.strip()]
                if lines:
                    tail = ("; backend stderr (last 20 lines):\n"
                            + "\n".join(lines[-20:]))
        except Exception:
            pass
        # ASTRA-Sim reports its own aborts on stdout, which read_wait has
        # already consumed; keep the last lines so the report says why.
        if out:
            last = [l.rstrip() for l in out[-8:] if l.strip()]
            if last:
                tail += "; backend stdout (last lines):\n" + "\n".join(last)
        raise RuntimeError(
            f"ASTRA-Sim backend terminated unexpectedly (exit code {rc}); "
            f"no further output will arrive{tail}")

    def write_flush(self, p, input):
        # For debugging
        # print(input)
        p.stdin.write(input+'\n')
        p.stdin.flush()
        return

    def parse_output(self, output):
        pattern = r"sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles, exposed communication (\d+) cycles."
        match = re.search(pattern, output)
        if match:
            sys = int(match.group(1))
            id = int(match.group(2))
            cycle = int(match.group(3))
            com_cycle = int(match.group(4))

            if self.end_dict[sys] != id:
                self.logger.info(
                    "NPU[%d] iteration %d finished, %d cycles, exposed communication %d cycles.",
                    sys,
                    id,
                    cycle,
                    com_cycle,
                )
                self.end_dict[sys] = id
            return {'sys': sys, 'id': id, 'cycle': cycle}
        return
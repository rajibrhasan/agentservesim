# The simulator image: ASTRA-Sim's toolchain plus the Python packages the
# engine imports. `scripts/docker-sim.sh` pip-installs these at container
# start, which is fine for an interactive shell but not for a policy search:
# every candidate evaluation would reinstall them.
FROM astrasim/tutorial-micro2024

RUN pip3 install --no-cache-dir \
        pyyaml pyinstrument transformers datasets msgspec \
        scikit-learn xgboost==3.1.2 matplotlib==3.5.3 \
        pandas==1.5.3 numpy==1.23.5

# The policy search runs in this image too, so a candidate is evaluated by
# calling the simulator in place (SIM_CONTAINER=none) rather than starting a
# container per candidate -- which would mean running docker from inside
# docker, and a second Python environment on the host to hold the search.
RUN pip3 install --no-cache-dir openevolve==0.2.11

WORKDIR /app/LLMServingSim

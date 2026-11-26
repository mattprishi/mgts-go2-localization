ARG BASE_IMAGE=hakatonius:latest
FROM ${BASE_IMAGE}

# The organizers' image already supplies NumPy, ROS 2 Humble and MCAP.
# Use a binary wheel for the image architecture and its CPython 3.10.
COPY scipy-*.whl /tmp/scipy-wheel/
RUN python3 -m pip install --no-index --no-deps /tmp/scipy-wheel/scipy-*.whl

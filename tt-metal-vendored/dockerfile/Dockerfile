# TT-Metalium Dockerfile

#############################################################

# Accept an argument to specify the Ubuntu version
ARG UBUNTU_VERSION=20.04
FROM public.ecr.aws/ubuntu/ubuntu:${UBUNTU_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime deps
COPY /install_dependencies.sh /opt/tt_metal_infra/scripts/docker/install_dependencies.sh
RUN /bin/bash /opt/tt_metal_infra/scripts/docker/install_dependencies.sh --docker --mode runtime


#############################################################

FROM base AS ci-build

RUN /bin/bash /opt/tt_metal_infra/scripts/docker/install_dependencies.sh --docker --mode build

# Install ccache from upstream; Apt's version for 20.04 predates remote_storage support
RUN mkdir -p /usr/local/bin && wget -O /tmp/ccache.tar.xz https://github.com/ccache/ccache/releases/download/v4.10.2/ccache-4.10.2-linux-x86_64.tar.xz && \
    tar -xf /tmp/ccache.tar.xz -C /usr/local/bin --strip-components=1 && \
    rm /tmp/ccache.tar.xz

ARG BOOST_VERSION=1.86.0
RUN mkdir -p /tmp/boost \
    && BOOST_VERSION_UNDERSCORE=$(echo ${BOOST_VERSION} | sed 's/\./_/g') \
    && wget -O /tmp/boost/boost_${BOOST_VERSION}.tar.gz "https://archives.boost.io/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_UNDERSCORE}.tar.gz" \
    && tar -xzf /tmp/boost/boost_${BOOST_VERSION}.tar.gz -C /tmp/boost --strip-components=1 \
    && cd /tmp/boost \
    && ./bootstrap.sh \
    && ./b2 install --prefix=/usr/local \
    && rm -rf /tmp/boost

ARG FMT_VERSION=11.0.1
RUN mkdir -p /tmp/fmt \
    && wget -O /tmp/fmt/fmt-${FMT_VERSION}.tar.gz "https://github.com/fmtlib/fmt/archive/${FMT_VERSION}.tar.gz" \
    && tar -xzf /tmp/fmt/fmt-${FMT_VERSION}.tar.gz -C /tmp/fmt --strip-components=1 \
    && cmake \
         -S /tmp/fmt \
         -B /tmp/fmt/build \
         -DCMAKE_BUILD_TYPE=Release \
         -DFMT_TEST=OFF \
         -DFMT_DOC=OFF \
         -DFMT_INSTALL=ON \
    && make -C /tmp/fmt/build -j$(nproc) \
    && make -C /tmp/fmt/build install \
    && rm -rf /tmp/fmt

ARG PYBIND11_VERSION=2.13.6
RUN mkdir -p /tmp/pybind11 \
    && wget -O /tmp/pybind11/pybind11-${PYBIND11_VERSION}.tar.gz "https://github.com/pybind/pybind11/archive/refs/tags/v${PYBIND11_VERSION}.tar.gz" \
    && tar -xzf /tmp/pybind11/pybind11-${PYBIND11_VERSION}.tar.gz -C /tmp/pybind11 --strip-components=1 \
    && cmake \
         -S /tmp/pybind11 \
         -B /tmp/pybind11/build \
         -DCMAKE_BUILD_TYPE=Release \
         -DPYBIND11_TEST=OFF \
         -DPYBIND11_INSTALL=ON \
    && make -C /tmp/pybind11/build -j$(nproc) \
    && make -C /tmp/pybind11/build install \
    && rm -rf /tmp/pybind11

ARG RANGE_V3_VERSION=0.12.0
RUN mkdir -p /tmp/range-v3 \
    && wget -O /tmp/range-v3/range-v3-${RANGE_V3_VERSION}.tar.gz "https://github.com/ericniebler/range-v3/archive/refs/tags/${RANGE_V3_VERSION}.tar.gz" \
    && tar -xzf /tmp/range-v3/range-v3-${RANGE_V3_VERSION}.tar.gz -C /tmp/range-v3 --strip-components=1 \
    && cmake \
         -S /tmp/range-v3 \
         -B /tmp/range-v3/build \
         -DCMAKE_BUILD_TYPE=Release \
         -DRANGE_V3_TESTS=OFF \
         -DRANGE_V3_EXAMPLES=OFF \
         -DRANGE_V3_DOCS=OFF \
         -DCMAKE_INSTALL_PREFIX=/usr/local \
    && make -C /tmp/range-v3/build -j$(nproc) \
    && make -C /tmp/range-v3/build install \
    && rm -rf /tmp/range-v3


# libstdc++ vs libc++ issue arises
#ARG YAML_VERSION=0.8.0
#RUN mkdir -p /tmp/yaml \
#    && wget -O /tmp/yaml/yaml-${YAML_VERSION}.tar.gz "https://github.com/jbeder/yaml-cpp/archive/refs/tags/${YAML_VERSION}.tar.gz" \
#    && tar -xzf /tmp/yaml/yaml-${YAML_VERSION}.tar.gz -C /tmp/yaml --strip-components=1 \
#    && cmake \
#         -S /tmp/yaml \
#         -B /tmp/yaml/build \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DYAML_CPP_BUILD_TESTS=OFF \
#         -DYAML_CPP_BUILD_TOOLS=OFF \
#         -DYAML_BUILD_SHARED_LIBS=OFF \
#    && make -C /tmp/yaml/build -j$(nproc) \
#    && make -C /tmp/yaml/build install \
#    && rm -rf /tmp/yaml

ARG JSON_VERSION=3.11.3
RUN mkdir -p /tmp/json \
    && wget -O /tmp/json/json-${JSON_VERSION}.tar.gz "https://github.com/nlohmann/json/archive/refs/tags/v${JSON_VERSION}.tar.gz" \
    && tar -xzf /tmp/json/json-${JSON_VERSION}.tar.gz -C /tmp/json --strip-components=1 \
    && cmake \
         -S /tmp/json \
         -B /tmp/json/build \
         -DCMAKE_BUILD_TYPE=Release \
         -DJSON_BuildTests=OFF \
         -DJSON_Install=ON \
    && make -C /tmp/json/build -j$(nproc) \
    && make -C /tmp/json/build install \
    && rm -rf /tmp/json

ARG MAGIC_ENUM_VERSION=0.9.7
RUN mkdir -p /tmp/magic_enum \
    && wget -O /tmp/magic_enum/magic_enum-${MAGIC_ENUM_VERSION}.tar.gz "https://github.com/Neargye/magic_enum/archive/refs/tags/v${MAGIC_ENUM_VERSION}.tar.gz" \
    && tar -xzf /tmp/magic_enum/magic_enum-${MAGIC_ENUM_VERSION}.tar.gz -C /tmp/magic_enum --strip-components=1 \
    && cmake \
         -S /tmp/magic_enum \
         -B /tmp/magic_enum/build \
         -DCMAKE_BUILD_TYPE=Release \
         -DMAGIC_ENUM_OPT_BUILD_TESTS=OFF \
         -DMAGIC_ENUM_OPT_BUILD_EXAMPLES=OFF \
         -DMAGIC_ENUM_OPT_INSTALL=ON \
    && make -C /tmp/magic_enum/build -j$(nproc) \
    && make -C /tmp/magic_enum/build install \
    && rm -rf /tmp/magic_enum

ARG TAKSFLOW_VERSION=3.7.0
RUN mkdir -p /tmp/taskflow \
    && wget -O /tmp/taskflow/taskflow-${TAKSFLOW_VERSION}.tar.gz "https://github.com/taskflow/taskflow/archive/v${TAKSFLOW_VERSION}.tar.gz" \
    && tar -xzf /tmp/taskflow/taskflow-${TAKSFLOW_VERSION}.tar.gz -C /tmp/taskflow --strip-components=1 \
    && cmake \
         -S /tmp/taskflow \
         -B /tmp/taskflow/build \
         -DCMAKE_BUILD_TYPE=Release \
         -DTF_BUILD_TESTS=OFF \
         -DTF_BUILD_EXAMPLES=OFF \
         -DTF_BUILD_BENCHMARKS=OFF \
         -DTF_BUILD_CUDA=OFF \
         -DTF_BUILD_SYCL=OFF \
    && make -C /tmp/taskflow/build -j$(nproc) \
    && make -C /tmp/taskflow/build install \
    && rm -rf /tmp/taskflow

ARG XTENSOR_XTL_VERSION=0.7.7
RUN mkdir -p /tmp/xtensor_xtl \
    && wget -O /tmp/xtensor_xtl/xtensor_xtl-${XTENSOR_XTL_VERSION}.tar.gz "https://github.com/xtensor-stack/xtl/archive/refs/tags/${XTENSOR_XTL_VERSION}.tar.gz" \
    && tar -xzf /tmp/xtensor_xtl/xtensor_xtl-${XTENSOR_XTL_VERSION}.tar.gz -C /tmp/xtensor_xtl --strip-components=1 \
    && cmake \
         -S /tmp/xtensor_xtl \
         -B /tmp/xtensor_xtl/build \
         -DCMAKE_BUILD_TYPE=Release \
    && make -C /tmp/xtensor_xtl/build -j$(nproc) \
    && make -C /tmp/xtensor_xtl/build install \
    && rm -rf /tmp/xtensor_xtl

# xtensor problemstic
#ARG XTENSOR_VERSION=0.25.0
#RUN mkdir -p /tmp/xtensor \
#    && wget -O /tmp/xtensor/xtensor-${XTENSOR_VERSION}.tar.gz "https://github.com/xtensor-stack/xtensor/archive/refs/tags/${XTENSOR_VERSION}.tar.gz" \
#    && tar -xzf /tmp/xtensor/xtensor-${XTENSOR_VERSION}.tar.gz -C /tmp/xtensor --strip-components=1 \
#    && cmake \
#         -S /tmp/xtensor \
#         -B /tmp/xtensor/build \
#         -DCMAKE_BUILD_TYPE=Release \
#    && make -C /tmp/xtensor/build -j$(nproc) \
#    && make -C /tmp/xtensor/build install \
#    && rm -rf /tmp/xtensor

# Issue arises - No blas
#ARG XTENSOR_BLAS_VERSION=0.21.0
#RUN mkdir -p /tmp/xtensor_blas \
#    && wget -O /tmp/xtensor_blas/xtensor_blas-${XTENSOR_BLAS_VERSION}.tar.gz "https://github.com/xtensor-stack/xtensor-blas/archive/refs/tags/${XTENSOR_BLAS_VERSION}.tar.gz" \
#    && tar -xzf /tmp/xtensor_blas/xtensor_blas-${XTENSOR_BLAS_VERSION}.tar.gz -C /tmp/xtensor_blas --strip-components=1 \
#    && cmake \
#         -S /tmp/xtensor_blas \
#         -B /tmp/xtensor_blas/build \
#         -DCMAKE_BUILD_TYPE=Release \
#    && make -C /tmp/xtensor_blas/build -j$(nproc) \
#    && make -C /tmp/xtensor_blas/build install \
#    && rm -rf /tmp/xtensor_blas

ARG DOXYGEN_VERSION=1.9.6
RUN mkdir -p /tmp/doxygen \
    && wget -O /tmp/doxygen/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz "https://www.doxygen.nl/files/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz" \
    && tar -xzf /tmp/doxygen/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz -C /tmp/doxygen --strip-components=1 \
    && make -C /tmp/doxygen -j$(nproc) \
    && make -C /tmp/doxygen install \
    && rm -rf /tmp/doxygen

RUN mkdir -p /tmp/cba \
    && wget -O /tmp/cba/cba.tar.gz https://github.com/aras-p/ClangBuildAnalyzer/archive/refs/tags/v1.6.0.tar.gz \
    && tar -xzf /tmp/cba/cba.tar.gz -C /tmp/cba --strip-components=1 \
    && cmake -S /tmp/cba/ -B /tmp/cba/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/cba/build \
    && cmake --install /tmp/cba/build \
    && rm -rf /tmp/cba

# Install extra ci apt requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    bc \
    clang-tidy-17 \
    curl \
    dialog \
    file \
    graphviz \
    jq \
    pandoc \
    sudo \
    wget \
    libtbb-dev \
    libcapstone-dev \
    libgtest-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV CCACHE_TEMPDIR=/tmp/ccache

#############################################################

FROM ci-build AS ci-test

ARG TT_METAL_INFRA_DIR=/opt/tt_metal_infra

# Create directories for infra
RUN mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/docs/
RUN mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/tests/sweep_framework/
RUN mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/
# Copy requirements from tt-metal folders with requirements.txt docs
COPY /docs/requirements-docs.txt ${TT_METAL_INFRA_DIR}/tt-metal/docs/.
# Copy requirements from tt-metal folders for sweeps (requirements-sweeps.txt)
COPY /tests/sweep_framework/requirements-sweeps.txt ${TT_METAL_INFRA_DIR}/tt-metal/tests/sweep_framework/.
COPY /tt_metal/python_env/requirements-dev.txt ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/.

RUN python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu && \
    python3 -m pip install setuptools wheel && \
    python3 -m pip install -r ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/requirements-dev.txt && \
    python3 -m pip install -r ${TT_METAL_INFRA_DIR}/tt-metal/docs/requirements-docs.txt

# Install test-infra specific apt requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    uuid-runtime \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

#############################################################

FROM ci-test AS dev

# Need this to build GDB
RUN apt-get -y update \
    && apt-get install -y libmpfr-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the gdb that is compatible with clang-17
RUN apt-get remove -y gdb || true \
    && mkdir -p /tmp/gdb-build && cd /tmp/gdb-build/ \
    && wget -O /tmp/gdb-build/gdb.tar.gz https://ftp.gnu.org/gnu/gdb/gdb-14.2.tar.gz \
    && tar -xvf /tmp/gdb-build/gdb.tar.gz -C /tmp/gdb-build --strip-components=1 \
    && /tmp/gdb-build/configure --prefix=/usr/local \
    && make -j$(nproc) \
    && make install \
    && rm -rf /tmp/gdb-build

# Install dev deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    acl \
    emacs \
    less \
    nano \
    openssh-server \
    vim \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

#############################################################

FROM dev AS release

RUN mkdir -p /etc && \
    echo "[global]\nextra-index-url = https://download.pytorch.org/whl/cpu" > /etc/pip.conf

ARG WHEEL_FILENAME
ADD $WHEEL_FILENAME $WHEEL_FILENAME
RUN pip3 install $WHEEL_FILENAME

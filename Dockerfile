FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        python3.10 python3.10-dev python3-pip tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install --no-cache-dir \
    torch==2.0.1+cu117 \
    --index-url https://download.pytorch.org/whl/cu117 && \
    pip install --no-cache-dir \
    numpy==1.24.4 \
    pandas==2.0.3 \
    scikit-learn==1.3.2 \
    tqdm==4.66.1 \
    pyarrow==14.0.1 \
    openpyxl==3.1.2 && \
    rm -rf /root/.cache/pip /tmp/* /var/tmp/* && \
    find /usr -name "*.pyc" -delete 2>/dev/null || true && \
    find /usr -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

WORKDIR /app

COPY app/ /app/
COPY test.sh /test.sh

RUN chmod +x /app/init.sh /app/train.sh /test.sh

CMD ["/bin/bash"]

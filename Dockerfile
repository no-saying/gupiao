FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        curl git tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy==1.24.4 \
    pandas==2.0.3 \
    scikit-learn==1.3.2 \
    tqdm==4.66.1 \
    pyarrow==14.0.1 \
    openpyxl==3.1.2

WORKDIR /app

COPY app/ /app/
COPY test.sh /test.sh

RUN chmod +x /app/init.sh /app/train.sh /test.sh

CMD ["/bin/bash"]

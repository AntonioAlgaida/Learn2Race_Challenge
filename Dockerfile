FROM aicrowd/learn-to-race:base

COPY apt.txt .
RUN apt -qq update && apt -qq install -y `cat apt.txt` \
 && rm -rf /var/cache/*

# Uncomment the lines to install required CUDA toolkit
# RUN conda install -y cudatoolkit=10.1
# RUN conda install -y cudatoolkit=10.2
# RUN conda install -y cudatoolkit=11.3

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


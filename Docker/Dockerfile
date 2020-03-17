# Matt Lee (lees4@ornl.gov)

FROM centos:centos7

EXPOSE 7777

RUN yum -y update && yum -y install tar && yum -y install wget && yum clean all
RUN yum -y install git
RUN yum -y install curl
RUN yum -y install bzip2
RUN curl -O https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
RUN bash Anaconda3-5.3.1-Linux-x86_64.sh -b -p /anaconda
RUN git clone https://github.com/ornlpmcp/ASCENDS.git
RUN echo 'export PATH="/anaconda/bin:$PATH"' >> ~/.bashrc 
RUN echo "source activate ascends" > ~/.bashrc
ENV PATH /anaconda/bin:$PATH
RUN /anaconda/bin/conda env create -f ASCENDS/environment.yml --name ascends 
CMD python /ASCENDS/train.py

#####################################
#
#         Docker file for SpaceSence
#         Base on GDAL image, ased on Ubuntu
#
####################################

FROM osgeo/gdal:ubuntu-small-latest
LABEL maintainer "Antoine Tavant <antoine.tavant@lpp.polytechnique.fr>"

# Update apt-get
RUN apt-get update && \
	apt-get install -qqy \
	git \
	wget \
	ipython3 \
	vim \
	python3-pip 

# Set default python to python3
# RUN rm /usr/bin/python /usr/bin/pip & ln -s /usr/bin/python3 /usr/bin/python & \
#    ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip3 install  numpy scipy pandas jupyter

# Settings for Jupyter-notebooks
EXPOSE 8888
ENTRYPOINT ["jupyter-notebook", "--allow-root", "--port=8888", "--no-browser", "--ip=0.0.0.0"] 



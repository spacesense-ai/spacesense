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
	vim \
	python3-pip \
	libsm6 \
	libxext6 \
	libxrender-dev

# Set default python to python3
# RUN rm /usr/bin/python /usr/bin/pip & ln -s /usr/bin/python3 /usr/bin/python & \
#    ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip3 install  numpy \
		scipy \
		pandas \
		ipython \
		jupyter \
		matplotlib \
		sentinelsat \
		sentinelhub \
		pyshp \
		pytest \
		sklearn \
		keras \
		tensorflow \
		geopandas \
		opencv-python\
		pyModis

RUN pip3 install python-cmr \
                 wget \
                 coverage \
                 appdirs \
                 rasterio

  
# Change the LANG to UTF8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Settings for Jupyter-notebooks
# EXPOSE 8888
# ENTRYPOINT ["jupyter-notebook", "--allow-root", "--port=8888", "--no-browser", "--ip=0.0.0.0"] 
RUN echo 'alias notebook="jupyter-notebook --allow-root --port=8888 --no-browser --ip=0.0.0.0"' >> ~/.bashrc

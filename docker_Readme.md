# What's up with the docker image ?

> The multiples dependances needed for spacesence can be difficult to intall on every system.
> Moreover, a rather new developpment method is the _shift left_, which prone to develop in an environment that is similare to the production.

## Build and use docker

A summary of the comment used:

1. First of all, you might need to start docker: `sudo systemctl start docker`

2. Build the image from the `Dockerfile`:
```
docker build -t spacesence/dev .
```

3. Run it !
```
docker run -it  spacesence/dev:latest
```
This will lauche the image.
The Entrypoint of the docker image is jupyter-notebook, exposed to the port `8888`.
You will need to upen it on you browser, and expose it when runing the image:
` docker run -it -p 8888:8888 spacesence/dev:latest`

	You can add a volume (like the folder where you work) with `-v <local path>:<path in docker>` 

4. Using jupyter
the docker image will lauch jupyter.
You will need to copy and past the URL given by jupyter, with its token.
It should looks like
```
http://127.0.0.1:8888/?token=b1903988532dce2f3f3682e9add6e56c1b7ef518122c04f4
```

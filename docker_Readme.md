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
You will need to upen it on you browser

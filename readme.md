# Style transfer app
Built using PyTorch and deployed using flask. Containerized. 

You can also run directly from source, provided you have all required dependencies.

View [demo](demo.mp4).

To build container use 
```bash
docker compose up --build
```
To run directly from source, make sure you have the dependencies installed, then use
```bash
flask run 
```

The number of iterations will reflect on the effects made on the original image. Please allow some time for the render to be completed. 


Demo was made using local dev environment with CUDA available, running from container will result in CUDA not being available, if you have a CUDA capable system, its recommended to run from source. 
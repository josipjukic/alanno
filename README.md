
![alanno_logo](https://user-images.githubusercontent.com/53186339/182347430-9d97ba90-adf2-4d3e-beae-ec72dbbf8d1b.png)

# Alanno - Active Learning Annotation
<br>
Annotation tool powered by Active Learning.
<br>

## Usage

The application can be run in two modes:

- Production
- Development

You will need [Docker](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/). 

First, you need to clone the repository:

```bash
$ git clone https://repos.takelab.fer.hr/jjukic/alanno.git
$ cd alanno
```

### Local:
```bash
$ docker-compose -f docker-compose.local.yml up
```

### Production:
```bash
$ docker-compose -f docker-compose.prod.yml up
```

Note : An example of relevant environmental variables is given in the `.env` file. You should replace the placeholder values.

Go to <http://localhost:8000/> to access the application.

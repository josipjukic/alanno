# Alanno - Active Learning Annotation
<br>
Annotation tool powered by Active Learning.
<br>

## Usage

The application can be run in two modes:

- Production
- Development

You will need Docker and Docker Compose. 

First, you need to clone the repository:

```bash
$ git clone https://repos.takelab.fer.hr/jjukic/alanno.git
$ cd alanno
```

### Production:
```bash
$ docker-compose -f docker-compose.prod.yml up
```

### Development:
```bash
$ docker-compose -f docker-compose.dev.yml up
```

Note : Export all relevant environmental variables for production setup.

Go to <http://localhost:8000/> to access the application.

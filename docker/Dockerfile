# Evaluator image: plain CPython, no extra packages. Evaluator code, candidate and
# instance files are bind-mounted read-only at run time; nothing is baked in.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /tmp
USER 65534:65534
CMD ["python", "-c", "print('evaluator image: mount code and run sandbox_entry.py')"]

container-full:
	docker build --platform linux/amd64 -t myapp . && docker run --rm myapp
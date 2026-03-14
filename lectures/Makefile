all:
	${MAKE} rm-pdf
	xelatex policies-value-functions.tex
	@if [ -f policies-value-functions.pdf ]; then \
		${MAKE} clean; \
		open policies-value-functions.pdf; \
	else \
		echo "Failed to generate PDF."; \
	fi
rm-pdf:
	rm -f *.pdf

clean:
	@rm -f *.aux *.log *.snm *.out *.toc *.nav

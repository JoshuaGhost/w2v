TEX=latex
TTP=texi2pdf
BIB=bibtex

dvi:
	$(TEX) acl2016.tex
	$(BIB) acl2016.aux
	$(BIB) acl2016.aux
	$(TEX) acl2016.tex

pdf:
	$(TTP) acl2016.tex
	$(BIB) acl2016.aux
	$(BIB) acl2016.aux
	$(TTP) acl2016.tex

clean:
	rm -f *.aux *.bbl *.blg *.dvi *.pdf *.log

.PHONY:
	dvi pdf clean

TEX=latex
TTP=texi2pdf
BIB=bibtex

dvi:
	$(TEX) acl2017.tex
	$(BIB) acl2017.aux
	$(BIB) acl2017.aux
	$(TEX) acl2017.tex

pdf:
	$(TTP) acl2017.tex
	$(BIB) acl2017.aux
	$(BIB) acl2017.aux
	$(TTP) acl2017.tex

clean:
	rm -f *.aux *.bbl *.blg *.dvi *.pdf *.log

.PHONY:
	dvi pdf clean

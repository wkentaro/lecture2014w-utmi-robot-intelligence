TEXFILE=$(wildcard *.tex)
DOCBASE=$(TEXFILE:.tex=)


doc:
	platex ${DOCBASE}.tex && dvipdfmx ${DOCBASE}.dvi
	platex ${DOCBASE}.tex && dvipdfmx ${DOCBASE}.dvi
	open ${DOCBASE}.pdf


commit:
	git add .
	git commit -m "Update report document"

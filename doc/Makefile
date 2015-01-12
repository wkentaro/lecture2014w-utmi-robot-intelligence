TEXFILE=$(wildcard *.tex)
DOCBASE=$(TEXFILE:.tex=)


pdf:
	platex ${DOCBASE}.tex && dvipdfmx ${DOCBASE}.dvi
	platex ${DOCBASE}.tex && dvipdfmx ${DOCBASE}.dvi
	open ${DOCBASE}.pdf


commit:
	git add .
	git commit -m "Update report document for `git log master -1 --pretty=short --abbrev-commit`" && git push origin master

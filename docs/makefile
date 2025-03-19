all: README.md build/t6alh.html build/docs.html
	cp README.md ../README.md
	cp build/t6alh.html ../../pnewstein.github.io/t6alh.html
	cp build/docs.html ../../pnewstein.github.io/t6alh-docs.html

build/%.html: %.md
	pandoc $< -o $@

README.md: README-template.md quickstart.md install.md
	xonsh make_readme.xon

build/t6alh.html: t6alh-template.html build/quickstart.html
	xonsh make_html.xsh

build/docs.md: docs-template.md ../template6alh/cli.py
	xonsh make_docs_md.xsh

build/docs.html: build/docs.md documentation-template.html
	pandoc build/docs.md --table-of-contents=true --toc-depth=4 -s -o build/docs.html --template=documentation-template.html



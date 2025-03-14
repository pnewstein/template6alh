all: README.md t6alh.html
	cp README.md ../README.md
	cp t6alh.html ../../pnewstein.github.io/t6alh.html

build/%.html: %.md
	pandoc $< -o $@

README.md: README-template.md quickstart.md install.md
	xonsh make_readme.xon

t6alh.html: t6alh-template.html build/quickstart.html
	xonsh make_html.xon

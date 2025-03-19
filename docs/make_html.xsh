quickstart = $(cat build/quickstart.html)
html = $(cat t6alh-template.html)
echo @(html.format(qickstart=quickstart)) > build/t6alh.html

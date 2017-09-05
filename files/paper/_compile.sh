rm paper.aux paper.b* paper.log
pdflatex paper.tex
bibtex paper.aux
pdflatex paper.tex

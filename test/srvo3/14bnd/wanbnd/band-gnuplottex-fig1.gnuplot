set terminal epslatex
set output 'band-gnuplottex-fig1.tex'
set size 0.618, 1.0
set xrange [0:0.66965]
set yrange [-8:6]

set xtics ("$\\mathrm{R}$" 0.0, "$\\Gamma$" 0.22538,"$\\mathrm{X}$" 0.35550 ,"$\\mathrm{M}$" 0.48563, "$\\Gamma$" 0.66965)
set ytics 2.0

set palette defined (1 "gray", 3 "red" )
unset colorbox

plot 'wanbnd.dat' u 1:2:3 w l lw 3 palette notitle


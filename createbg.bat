@echo off

setlocal EnableDelayedExpansion

set/p U=Cual es la direccion de las imagenes?
pause
for /R %U% %%i in ("*") do echo %%i >> bg.txt
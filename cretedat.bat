@echo off

setlocal EnableDelayedExpansion
set size= 1 0 0 38 46
set/p U=Cual es la direccion de las imagenes?
set sdir=/neg
pause
for /R %U% %%i in ("*") do echo %%i%size% >> product.info
## Copyright (C) 2016 Johannes
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{retval} =} mit_horner (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Johannes <Johannes@YOHIBOOK>
## Created: 2016-05-04

function [retval] = mit_horner (input1, input2)
x = linspace(0.8,1.2, 5*10^(5));
y = (((((((single(x)).-7).*(single(x)).+21).*(single(x)).-35).*(single(x)).+35).*(single(x)).-21).*(single(x)).+7).*(single(x)).-1;
plot(x,y)
endfunction

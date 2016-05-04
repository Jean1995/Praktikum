function a = euler1(value,n);
%euler(value,n) gibt die normale Summation wieder, wobei value fuer den x Wert
%und n fuer die Summationsgroesse steht
if n < 0;
  printf("negative n, pls enter a positive value");
else;
  for i=0:1:n;
    z = 1;
    if i == 0;
      a = 1;
    else;
      for b = 1:1:i;
        z = z*b;
      end;
    a = a + (value)^i/z;
    end;
  end;
end;
endfunction;


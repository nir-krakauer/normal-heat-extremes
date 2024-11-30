function [nlogL, k_terms] = gevnll (x, k, sigma, mu)
#calculate GEV negative log likelihood
#no input checking done

  k_terms = [];
  a = (x - mu) ./ sigma;

  if all(k == 0)
    nlogL = exp(-a) + a + log(sigma);
  else
    aa = k .* a;
    if min(abs(aa)) < 1E-4 && max(abs(aa)) < 0.5 #use a series expansion to find the log likelihood more accurately when k is small
      k_terms = 1; sgn = 1; i = 0;
      while 1
        sgn = -sgn; i++;
        newterm = sgn * (aa .^ i) / (i + 1);
        k_terms = k_terms + newterm;
        if max(abs(newterm)) <= eps
          break
        endif
      endwhile
      nlogL = exp(-a .* k_terms) + a .* (k + 1) .* k_terms + log(sigma);
    else
      b = 1 + aa;
      nlogL = b .^ (-1 ./ k) + (1 + 1 ./ k) .* log(b) + log(sigma);
      nlogL(b <= 0) = Inf;
    endif
  endif

endfunction



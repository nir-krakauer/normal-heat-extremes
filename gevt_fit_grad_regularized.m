## Copyright (C) 2024 Nir Krakauer <nkrakauer@ccny.cuny.edu>
##
## This program is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License as published by the Free Software
## Foundation; either version 3 of the License, or (at your option) any later
## version.
##
## This program is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
## FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along with
## this program; if not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {[@var{paramhat}, @var{FIM}, @var{warnflag}] =} gevt_fit_grad_regularized  (@var{data}, @var{z})
## @deftypefnx {[@var{paramhat}, @var{FIM}, @var{warnflag}] =} gevt_fit_grad_regularized  (@var{data}, @var{z}, @var{paramguess}, @var{l})
##
## Find the maximum likelihood estimator @var{paramhat} of the generalized
## extreme value (GEV) distribution with trend in the location parameter to fit @var{data}
## using the analytic gradient.
##
## Adds a quadratic panelty term to the cost function
## to keep the shape parameter close to zero
## for small sample sizes.
##
## Calls fminunc from the Optimization package.
##
## @subheading Arguments
##
## @itemize @bullet
## @item
## @var{data} is the vector of given values.
## @item
## @var{z} is the vector of covariate values, same size as @var{data}. 
## The location parameter is mu0 + mu1*z.
## @item
## @var{paramguess} is an initial guess for the maximum likelihood parameter
## vector. If not given or left empty, this defaults to @var{k_0}=0 and @var{sigma}, @var{mu}
## determined by matching the data mean and standard deviation to their expected
## values.
## @item
## @var{l} is a nonnegative scalar, controls amount of regularization. (0 = no regularization is the default)
## @end itemize
##
## @subheading Return values
##
## @itemize @bullet
## @item
## @var{paramhat} is the 4-parameter maximum-likelihood parameter vector
## @code{[@var{k_0}, @var{sigma}, @var{mu0}, @var{mu1}]}, where @var{k_0} is the shape
## parameter of the GEV distribution, @var{sigma} is the scale parameter of the
## GEV distribution, and @var{mu}=@var{mu0} + @var{mu1}*@var{z} is the location parameter of the GEV
## distribution.
## @item
## @var{FIM} has the Fisher information matrix at the maximum-likelihood
## position.
## @item
## @var{warnflag} is set to true if optimum parameter values do not appear to have been reached.
## One possible mitigation is to increase @var{l}.
## @end itemize
##
## When K < 0, the GEV is the type III extreme value distribution.  When K > 0,
## the GEV distribution is the type II, or Frechet, extreme value distribution.
## If W has a Weibull distribution as computed by the WBLFIT function, then -W
## has a type III extreme value distribution and 1/W has a type II extreme value
## distribution.  In the limit as K approaches 0, the GEV is the mirror image of
## the type I extreme value distribution as computed by the EVFIT function.
##
## The mean of the GEV distribution is not finite when K >= 1, and the variance
## is not finite when K >= 1/2.  The GEV distribution is defined
## for K*(X-MU)/SIGMA > -1.
##
## @subheading Examples
##
## @example
## @group
## data = 1:50;
## [pfit, pci] = gevfit_grad (data);
## p1 = gevcdf (data, pfit(1), pfit(2), pfit(3));
## plot (data, p1)
## @end group
## @end example
## @seealso{gevt_fit_grad}
## @end deftypefn

function [paramhat, FIM, warnflag] = gevt_fit_grad_regularized (data, z, paramguess, l)

  ## Check arguments
  if (nargin < 2)
    print_usage;
  endif
  
  
  ## Check data is vector
  if (! isvector (data))
    error ("gevt_fit_grad_regularized: DATA must be a vector");
  endif

  ## Check second input argument
  ## If it is a scalar, then it is ALPHA value,
  ## otherwise it is an initial parameter vector
  if (nargin <= 2)
    paramguess = [];
  endif
  if (nargin <= 3) || isempty(l)
    l = 0;
  endif
  warnflag = false;

## Compute initial parameters if not parsed as an input argument
  if (isempty (paramguess))
    paramguess = [gevfit_lmom(data)', 0];
    k_0 = paramguess(1);
    ## Check if data support initial parameters or fall back to unbounded evfit
    if (k_0 < 0 && (max (data) > - paramguess(2) / k_0 + paramguess(3)) || ...
        k_0 > 0 && (min (data) < - paramguess(2) / k_0 + paramguess(3)))
      paramguess = [0, evfit(data), 0];
      paramguess = flip (paramguess);
    endif
  endif
  options.Display = "off";
  options.MaxFunEvals = 4000;
  options.MaxIter = 1000;
  options.TolFun = 1e-12;
  options.TolX = 1e-12;
  options.GradObj = "on";
  
  ## Minimize the negative log-likelihood according to initial parameters
  paramguess(2) = log (paramguess(2));
  fhandle = @(paramguess) nll_and_grad (paramguess, data, z, l);
  
  [paramhat, ~, exitflag, output] = fminunc (fhandle, paramguess, options);
  #paramguess
  #[NLL, GRAD] = nll_and_grad (paramguess, data, z, l)
  #[NLL, GRAD] = nll_and_grad (paramhat, data, z, l)
  #exitflag, output
  #paramhat
  paramhat(2) = exp (paramhat(2));
  ## Display errors and warnings if any
  if (exitflag == 0)
    if (output.funcCount >= output.iterations)
      warning ("gevt_fit_grad_regularized: maximum number of evaluations reached");
    else
      warning ("gevt_fit_grad_regularized: reached iteration limit");
    endif
  elseif (exitflag == -1)
    error ("gevt_fit_grad_regularized: No solution");
  endif
  ## Return a row vector for Matlab compatibility
  paramhat = paramhat(:)';
  ## Check for second output argument
  if (nargout > 1)
  	[~, GRAD, FIM] = gevt_like (paramhat, data, z);
  	FIM(1, 1) += 2*l;
  	GRAD(1) += 2 * l * paramhat(1);
  	if max(abs(GRAD)) > 1E-3
  	  warning ("gevt_fit_grad_regularized:gradient", "gevt_fit_grad_regularized: gradient of cost function is not close to zero, solution may not have been reached")
  	  warnflag = true;
  	  disp(GRAD)
    endif
  endif
endfunction


function [NLL, GRAD] = nll_and_grad (parms, data, z, l)
  [NLL, k_terms] = nll (parms, data, z);
  
  NLL = sum (NLL) + l * parms(1)^2;

  if (nargout > 1)
    k = parms(1);
    log_sigma = parms(2);
    sigma = exp (log_sigma);
    mu0 = parms(3);
    mu1 = parms(4);
    [GRAD, kk_terms] = gevgrad (data, k, sigma, mu0, mu1, z, k_terms);
    GRAD(1) += 2 * l * parms(1);
  endif

endfunction

## Negative log-likelihood for the GEV (log(sigma) parameterization)
function [nlogL, k_terms] = nll (parms, data, z)
  k = parms(1);
  log_sigma = parms(2);
  sigma = exp (log_sigma);
  mu = parms(3) + parms(4)*z;
  
  n = numel (data);

  k_terms = [];
  a = (data - mu) ./ sigma;

  if (all (k == 0))
    nlogL = exp(-a) + a + log_sigma;
  else
    aa = k .* a;
    ## Use a series expansion to find the log likelihood more accurately
    ## when k is small
    if (min (abs (aa)) < 1E-3 && max (abs (aa)) < 0.5)
      k_terms = 1;
      sgn = 1;
      i = 0;
      while 1
        sgn = -sgn;
        i++;
        newterm = (sgn  / (i + 1)) * (aa .^ i);
        k_terms = k_terms + newterm;
        if (max (abs (newterm)) <= eps)
          break
        endif
      endwhile
      nlogL = exp (-a .* k_terms) + a .* (k + 1) .* k_terms + log_sigma;
    else
      b = 1 + aa;
      nlogL = b .^ (-1 ./ k) + (1 + 1 ./ k) .* log (b) + log_sigma;
      nlogL(b <= 0) = Inf;
    endif
  endif
endfunction

## NLL gradient for the GEV (log(sigma) parameterization) 
function [G, kk_terms] = gevgrad (x, k, sigma, mu0, mu1, z, k_terms)

  mu = mu0 + mu1*z;

  kk_terms = [];
  G = ones(1, 4);
  ## Use the expressions for first derivatives that are the limits as k --> 0
  if (k == 0)
    a = (x - mu) ./ sigma;
    f = exp(-a) - 1;
    ## k
    g = a .* (1 + a .* f / 2);
    G(1) = sum(g(:));
    ## sigma
    g = (a .* f + 1) ./ sigma;
    G(2) = sum(g(:));
    ## mu
    g = f ./ sigma;
    G(3) = sum(g(:));
    G(4) = sum((g .* z)(:));
    return
  endif

  a = (x - mu) ./ sigma;
  b = 1 + k .* a;
  ## Negative log likelihood is locally infinite
  if (any (b <= 0))
    G(:) = 0;
    return
  endif
  ## k
  c = log(b);
  d = 1 ./ k + 1;
  ## Use a series expansion to find the gradient more accurately when k is small
  if (nargin > 4 && ! isempty (k_terms))
    aa = k .* a;
    f = exp (-a .* k_terms);
    kk_terms = 0.5;
    sgn = 1;
    i = 0;
    while 1
      sgn = -sgn;
      i++;
      newterm = (sgn * (i + 1) / (i + 2)) * (aa .^ i);
      kk_terms = kk_terms + newterm;
      if (max (abs (newterm)) <= eps)
        break
      endif
    endwhile
    g = a .* ((a .* kk_terms) .* (f - 1 - k) + k_terms);
  else
    g = (c ./ k - a ./ b) ./ (k .* b .^ (1/k)) - c ./ (k .^ 2) + a .* d ./ b;
  endif
  G(1) = sum(g(:));

  ## sigma
  ## Use a series expansion to find the gradient more accurately when k is small
  if nargin > 4 && ~isempty(k_terms)
    g = (1 - a .* (a .* k .* kk_terms - k_terms) .* (f - k - 1));
  else
    g = (a .* b .^ (-d) - (k + 1) .* a ./ b + 1);
  endif
  G(2) = sum(g(:));

  ## mu
  ## Use a series expansion to find the gradient more accurately when k is small
  if (nargin > 4 && ! isempty (k_terms))
    g = - (a .* k .* kk_terms - k_terms) .* (f - k - 1) ./ sigma;
  else
    g = (b .^ (-d) - (k + 1) ./ b) ./ sigma;
  end
  G(3) = sum(g(:));
  
  G(4) = sum((g .* z)(:));

endfunction

%!demo
%! data = 1:50;
%! [pfit, pci] = gevfit_grad (data);
%! p1 = gevcdf (data, pfit(1), pfit(2), pfit(3));
%! plot (data, p1);

%!test
%! data = 1:50;
%! [pfit, pci] = gevfit_grad (data);
%! pfit_out = [-0.4407, 15.1923, 21.5309];
%! pci_out = [-0.7532, 11.5878, 16.5686; -0.1282, 19.9183, 26.4926];
%! assert (pfit, pfit_out, 1e-3);
%! assert (pci, pci_out, 1e-3);
%!error [pfit, pci] = gevfit_grad (ones (2,5));

%!test
%! data = 1:2:50;
%! [pfit, pci] = gevfit_grad (data);
%! pfit_out = [-0.4434, 15.2024, 21.0532];
%! pci_out = [-0.8904, 10.3439, 14.0168; 0.0035, 22.3429, 28.0896];
%! assert (pfit, pfit_out, 1e-3);
%! assert (pci, pci_out, 1e-3);

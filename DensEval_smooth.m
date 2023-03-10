function [p,x,yp,ypo] = DensEval_smooth(t,RC)
%DensEval evaluates the log density p at t
%
psi1 = t(1);
psi2 = t(2);
%
Sig_eps = exp(2*psi1)*sparse(eye(RC.n));
Sig_x = [RC.sigmab,zeros(1,RC.n);zeros(RC.n,1),exp(2*psi2)*RC.R];
%
L = chol(full(RC.Z*Sig_x*RC.Z' + Sig_eps + 10^(-10)*sparse(eye(RC.n))))';
w = L\(RC.y-RC.Z*RC.mu_x);
%
p = -0.5*w'*w-sum(log(diag(L))) ...            % log(p(y*|t))
    + psi1 - exp(psi1)*RC.lam_psi1 ...            % log(p(psi1))  
    + psi2 - exp(psi2)*RC.lam_psi2;         % log(p(psi2))
%
W = L\(RC.Z*Sig_x);
x_u = RC.mu_x + chol(Sig_x)'*normrnd(0,1,[RC.n+1,1]);
sss = (RC.Z*x_u - RC.y + exp(psi1)*normrnd(0,1,[RC.n,1]));
x = x_u - W'*(L\(sss));
%
yp = RC.Z*x;
yp = yp(1:RC.n,:);
ypo = yp+exp(psi1)*normrnd(0,1,[RC.n,1]);
end
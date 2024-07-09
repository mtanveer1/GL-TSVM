function[k]=kernelfunction(type,u,v,p)

% Function to compute kernel
% This function computes linear and RBF kernels
% Inputs:
% type: Kernel type (1:linear, 2:RBF
switch type
    
    case 1;
        %Linear Kernal
        k = u*v';
        
    case 2;
        %Radial Basia Function Kernal
        k = exp(-(u-v)*(u-v)'/(p.^2));
        
    otherwise
        k=0;
        
end
return

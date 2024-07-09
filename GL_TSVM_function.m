function [Accuracy]=GL_TSVM_function(xTrain,yTrain,xTest,yTest,a,C,c,sigma)

% Separate data of the two classes
A=xTrain(yTrain==1,:);
B=xTrain(yTrain==-1,:);
kernel_type=2;


% Compute training and test kernels
for i=1:size(A,1)
    for j=1:size(xTrain,1)
        KAC(i,j)=kernelfunction(kernel_type,A(i,:),xTrain(j,:),sigma);
    end
end

for i=1:size(B,1)
    for j=1:size(xTrain,1)
        KBC(i,j)=kernelfunction(kernel_type,B(i,:),xTrain(j,:),sigma);
    end
end

for i=1:size(xTest,1)
    for j=1:size(xTrain,1)
        KTest(i,j)=kernelfunction(kernel_type,xTest(i,:),xTrain(j,:),sigma);
    end
end
n1=size(KAC,2);
n2=size(KBC,2);
n3=size(KAC,1);
n4=size(KBC,1);


function f1_Z1=gfun1(Z1)
R2=zeros(size(KBC,2),size(KBC,1));
for j=1:size(KBC,1)

R2(:,j)=c.*KBC(j,:)' .* (   ( ( a*(1+KBC(j,:)*Z1)+1 ) * exp(a*(1+KBC(j,:)*Z1)) -1 ) / ( 1 + (1+KBC(j,:)*Z1)  *  (exp(a*(1+KBC(j,:)*Z1))-1) )^2   ) ;     
end
f1_Z1=sum(R2,2);
end



function f2_Z2=gfun2(Z2) 
S2=zeros(size(KAC,2),size(KAC,1));
for j=1:size(KAC,1)

S2(:,j)= -c.*KAC(j,:)'.*    (   ( ( a*(1-KAC(j,:)*Z2)+1 ) * exp(a*(1-KAC(j,:)*Z2)) -1 ) / ( 1 + (1-KAC(j,:)*Z2)  *  (exp(a*(1-KAC(j,:)*Z2))-1) )^2   ) ;           
end
f2_Z2=sum(S2,2);
end
N=50;
t=1;
Z10=zeros(size(KAC,2),1);
Z20=zeros(size(KBC,2),1);

while(t<N)   
 g1t=feval(@gfun1,Z10);   
 Z1=1/C*(eye(n1)-KAC'*((C*eye(n3)+KAC*KAC')\KAC))*g1t;
    if(norm(Z1-Z10)<1e-6)
        break;
    end
    Z10=Z1;
    t=t+1;
end
uu1=Z1(1:(size(Z1,1)-1),1);
bb1=Z1(end,1);
t=1;
while(t<N)
    g2t=feval(@gfun2,Z20);   
    Z2=-1/C*(eye(n2)-KBC'*((C*eye(n4)+KBC*KBC')\KBC))*g2t;
    if(norm(Z2-Z20)<1e-6)
        break;
    end
   
    Z20=Z2;
    t=t+1;
end
uu2=Z2(1:(size(Z2,1)-1),1);
bb2=Z2(end,1);

u1=[uu1;bb1];
u2=[uu2;bb2];
d1=abs(KTest*u1);
d2=abs(KTest*u2);
y=d1-d2;
y(y<0)=1;
y(y~=1)=-1;
preY=y;
err=sum(preY~=yTest)/size(KTest,1);
Accuracy=1-err;

end

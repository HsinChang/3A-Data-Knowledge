clear 
close all
clc

load('uclaf_data.mat');

beta = [1.5, 1.5, 1.5, 0.3, 0.3]; %beta for every tensor
w = [1, 0.1, 0.1, 0.1, 0.1]; %the weight of each tensor for the loss function

%alpha = 0.00001;
k = 3;
nIter = 2;

U = rand(size(UserLocAct, 1),k);
L = rand(size(UserLocAct, 2),k);
A = rand(size(UserLocAct, 3),k);
F = rand(size(LocFea, 2), k);
U2 = rand(k, size(UserLocAct, 1));
A2 = rand(k, size(UserLocAct, 3));
L2 = rand(k, size(UserLocAct, 2));

ULAhat = zeros(size(UserLocAct));
for j = 1:size(UserLocAct, 3),  ULAhat(:,:,j) = U * diag(A(j,:)) * L'; end 

loss1 = sum(sum(sum(UserLocAct.^(beta(1))./(beta(1).*(beta(1)-1)) - UserLocAct.*(ULAhat.^(beta(1)-1))./(beta(1)-1)+ULAhat.^(beta(1))./beta(1))));
loss2 = sum(sum(UserLoc.^(beta(2))./(beta(2).*(beta(2)-1))-UserLoc.*((U*L2).^(beta(2)-1))./(beta(2)-1)+((U*L2).^(beta(2)))./beta(2)));
loss3 = sum(sum(LocFea.^(beta(3))./(beta(3).*(beta(3)-1))-LocFea.*((L*F').^(beta(3)-1))./(beta(3)-1)+((L*F').^(beta(3)))./beta(3)));
loss4 = sum(sum(UserUser.^(beta(4))./(beta(4).*(beta(4)-1))-UserUser.*((U*U2).^(beta(4)-1))./(beta(4)-1)+((U*U2).^(beta(4)))./beta(4)));
loss5 = sum(sum(ActAct.^(beta(5))./(beta(5).*(beta(5)-1))-ActAct.*((A*A2).^(beta(5)-1))./(beta(5)-1)+((A*A2).^(beta(5)))./beta(5)));
Loss = abs(loss1+w(2)*loss2+w(3)*loss3+w(4)*loss4+w(5)*loss5);
oldLoss = Loss;
obj = zeros(1,nIter);

for it = 1:nIter

    U = U.*(((tenmat(UserLocAct.*(ULAhat.^(beta(1)-2)),1)*khatrirao(A,L)).data + w(2)*UserLoc.*((U*L2).^(beta(2)-2))*L2'+w(4)*UserUser.*((U*U2).^(beta(4)-2))*U2')./((tenmat(ULAhat.^(beta(1)-1),1)*khatrirao(A,L)).data+w(2)*(U*L2).^(beta(2)-1)*L2'+w(4)*(U*U2).^(beta(4)-1)*U2'));
    L = L.*(((tenmat(UserLocAct.*(ULAhat.^(beta(1)-2)),2)*khatrirao(A,U)).data + w(3)*LocFea.*((L*F').^(beta(3)-2))*F)./((tenmat(ULAhat.^(beta(1)-1),2)*khatrirao(A,U)).data+w(3)*(L*F').^(beta(3)-1)*F));
    A = A.*(((tenmat(UserLocAct.*(ULAhat.^(beta(1)-2)),3)*khatrirao(L,U)).data + w(5)*ActAct.*((A*A2).^(beta(5)-2))*A2')./((tenmat(ULAhat.^(beta(1)-1),3)*khatrirao(L,U)).data+w(5)*(A*A2).^(beta(5)-1)*A2'));
    F = F.*(w(3)*(LocFea'*L)./((L*F')'*L));
    U2 = U2.*(w(4)*(U'*UserUser)./(U'*U*U2));
    L2 = L2.*(w(2)*(U'*UserLoc)./(U'*U*L2));
    A2 = A2.*(w(5)*(A'*ActAct)./(A'*A*A2));

    for j = 1:size(UserLocAct, 3),  ULAhat(:,:,j) = U * diag(A(j,:)) * L'; end 

	loss1 = sum(sum(sum(UserLocAct.^(beta(1))./(beta(1).*(beta(1)-1)) - UserLocAct.*(ULAhat.^(beta(1)-1))./(beta(1)-1)+ULAhat.^(beta(1))./beta(1))));
	loss2 = sum(sum(UserLoc.^(beta(2))./(beta(2).*(beta(2)-1))-UserLoc.*((U*L2).^(beta(2)-1))./(beta(2)-1)+((U*L2).^(beta(2)))./beta(2)));
	loss3 = sum(sum(LocFea.^(beta(3))./(beta(3).*(beta(3)-1))-LocFea.*((L*F').^(beta(3)-1))./(beta(3)-1)+((L*F').^(beta(3)))./beta(3)));
	loss4 = sum(sum(UserUser.^(beta(4))./(beta(4).*(beta(4)-1))-UserUser.*((U*U2).^(beta(4)-1))./(beta(4)-1)+((U*U2).^(beta(4)))./beta(4)));
	loss5 = sum(sum(ActAct.^(beta(5))./(beta(5).*(beta(5)-1))-ActAct.*((A*A2).^(beta(5)-1))./(beta(5)-1)+((A*A2).^(beta(5)))./beta(5)));
	Loss = loss1+w(2)*loss2+w(3)*loss3+w(4)*loss4+w(5)*loss5;
	obj(it) = Loss;

end

figure,
plot(obj);
currentFun = @currentPFet;

src = 0.0:0.01:1.8;
gate = 0.0:0.01:1.8;
%gate = 0.6;
%drain = 0.0:0.01:1.8;
drain = 0.8
%drain = 0.0;

Vin = src;
Vout = gate;
%Vout = drain;

I = zeros(length(Vout), length(Vin));
firDerSrc = zeros(length(Vout), length(Vin));
firDerGate = zeros(length(Vout), length(Vin));
firDerDrain = zeros(length(Vout), length(Vin));
secDerSrc = zeros(length(Vout), length(Vin));
secDerGate = zeros(length(Vout), length(Vin));
secDerDrain = zeros(length(Vout), length(Vin));
secDerSrcGate = zeros(length(Vout), length(Vin));
secDerSrcDrain = zeros(length(Vout), length(Vin));
secDerGateDrain = zeros(length(Vout), length(Vin));
for i = 1:length(Vin)
	for j = 1:length(Vout)
		%[I(j,i), firDerSrc(j,i), firDerGrnd(j,i), firDerDrain(j,i), secDerSrc(j,i), secDerGrnd(j,i), secDerDrain(j,i), secDerSrcGrnd(j,i), secDerSrcDrain(j,i), secDerGrndDrain(j,i)] = currentFun(Vin(i), gate, Vout(j));
		[I(j,i), firDerSrc(j,i), firDerGate(j,i), firDerDrain(j,i), secDerSrc(j,i), secDerGate(j,i), secDerDrain(j,i), secDerSrcGate(j,i), secDerSrcDrain(j,i), secDerGateDrain(j,i)] = currentFun(Vin(i), Vout(j), drain);
	end;
end;

secDerIn = secDerSrc;
secDerOut = secDerGate;
%secDerOut = secDerDrain;

[surfVin,surfVout] = meshgrid(Vin,Vout);
I;

[posSecDerInR, posSecDerInC] = find(secDerIn > 0);
[negSecDerInR, negSecDerInC] = find(secDerIn < 0);

[posSecDerOutR, posSecDerOutC] = find(secDerOut > 0);
[negSecDerOutR, negSecDerOutC] = find(secDerOut < 0);

%[posSecDerOutInR, posSecDerOutInC] = find(secDerOutIn > 0);
%[negSecDerOutInR, negSecDerOutInC] = find(secDerOutIn < 0);

CO(:,:,1) = zeros(length(Vout), length(Vin));
CO(:,:,2) = zeros(length(Vout), length(Vin));
CO(:,:,3) = zeros(length(Vout), length(Vin));

for i = 1:length(posSecDerInR)
	%CO(posSecDerInR(i),posSecDerInC(i),1) = 0.0; 
	CO(posSecDerInR(i),posSecDerInC(i),2) = 1.0; 
	%CO(posSecDerInR(i),posSecDerInC(i),3) = 0.0; 
end;

for i = 1:length(negSecDerInR)
	%CO(negSecDerInR(i),negSecDerInC(i),1) = 0.0; 
	CO(negSecDerInR(i),negSecDerInC(i),2) = 1.0; 
	%CO(negSecDerInR(i),negSecDerInC(i),3) = 0.0; 
end;

for i = 1:length(posSecDerOutR)
	%CO(negSecDerOutR(i),negSecDerOutC(i),1) = 0.0; 
	%CO(negSecDerOutR(i),negSecDerOutC(i),2) = 0.0; 
	CO(posSecDerOutR(i),posSecDerOutC(i),3) = 1.0;  
end;

for i = 1:length(negSecDerOutR)
	%CO(negSecDerOutR(i),negSecDerOutC(i),1) = 0.0; 
	%CO(negSecDerOutR(i),negSecDerOutC(i),2) = 0.0; 
	CO(negSecDerOutR(i),negSecDerOutC(i),3) = 1.0;  
end;

%for i = 1:length(posSecDerOutInR)
%	%CO(posSecDerInR(i),posSecDerInC(i),1) = 0.0; 
%	CO(posSecDerOutInR(i),posSecDerOutInC(i),2) = 1.0; 
%	%CO(posSecDerInR(i),posSecDerInC(i),3) = 0.0; 
%end;

%for i = 1:length(negSecDerOutInR)
%	%CO(negSecDerInR(i),negSecDerInC(i),1) = 0.0; 
%	%CO(negSecDerOutInR(i),negSecDerOutInC(i),2) = 1.0; 
%	CO(negSecDerOutInR(i),negSecDerOutInC(i),3) = 1.0;
%end;


figure;
s1 = surf(surfVin, surfVout, I, CO);
shading interp;
alpha(s1, 0.5)
%alpha(cHull1, 0.5)
%alpha(cHull2, 0.5)
xlabel('Vin')
ylabel('Vout')

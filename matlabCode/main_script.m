% Itrat Ahmed Akhter
% CPSC 538G Proposal
% main_script.m
% figures for project report

%Figure 2
%Vin = 0.0:0.01:1;
%figure;
%for Vout = 0.0:0.25:1.0;
%	currents = zeros(size(Vin));
%	for idx = 1:numel(Vin)
%		I = currentMosfet(Vin(idx),Vout);
%		currents(idx) = I;
%	end;
%	plot(Vin, currents);
%	hold on;
%end;
%xlabel('Vin')
%ylabel('Current')
%legend('0.0', '0.25', '0.50','0.75','1.0')

%Vout = 0.0:0.01:1;
%figure;
%for Vin = 0.0:0.25:1.0;
%	currents = zeros(size(Vout));
%	for idx = 1:numel(Vout)
%		I = currentMosfet(Vin,Vout(idx));
%		currents(idx) = I;
%	end;
%	plot(Vout, currents);
%	hold on;
%end;
%xlabel('Vout')
%ylabel('Current')
%legend('0.0', '0.25', '0.50','0.75','1.0')

currentFun = @currentMosfet;
%currentFun = @exampleFun;

%hyperIn = [0.85; 0.90; 0.85; 0.90];
%hyperOut = [0.93; 0.93; 0.95; 0.95];

%hyperIn = [0.85; 0.90; 0.85; 0.90];
%hyperOut = [0.85; 0.85; 0.95; 0.95];

hyperIn = [0.75; 1.0; 0.75; 1.0];
hyperOut = [0.75; 0.75; 1.0; 1.0];

hyperIn = [0.0; 0.0; 1.0; 1.0];
hyperOut = [0.0; 1.0; 0.0; 1.0];

Vin = 0.0:0.01:1.0;
Vout = 0.0:0.01:1.0;
%Vin = min(hyperIn):0.01:max(hyperIn);
%Vout = min(hyperOut):0.01:max(hyperOut);
I = zeros(length(Vout), length(Vin));
firDerIn = zeros(length(Vout), length(Vin));
firDerOut = zeros(length(Vout), length(Vin));
secDerIn = zeros(length(Vout), length(Vin));
secDerOut = zeros(length(Vout), length(Vin));
for i = 1:length(Vin)
	for j = 1:length(Vout)
		[I(j,i), firDerIn(j,i), firDerOut(j,i), secDerIn(j,i), secDerOut(j,i)] = currentFun(Vin(i), Vout(j));
	end;
end;
[surfVin,surfVout] = meshgrid(Vin,Vout);
I;

[posSecDerInR, posSecDerInC] = find(secDerIn > 0);
[negSecDerInR, negSecDerInC] = find(secDerIn < 0);

[posSecDerOutR, posSecDerOutC] = find(secDerOut > 0);
[negSecDerOutR, negSecDerOutC] = find(secDerOut < 0);

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

currentsAtHypers = zeros(length(hyperIn));
firDerInAtHypers = zeros(length(hyperIn));
firDerOutAtHypers = zeros(length(hyperIn));
for i = 1:length(hyperIn)
	[currentAtHypers(i),firDerInAtHypers(i),firDerOutAtHypers(i),blah3,blah4] = currentFun(hyperIn(i), hyperOut(i));
end;
currentAtHypers
planeVin = min(hyperIn):0.01:max(hyperIn);
planeVout = min(hyperOut):0.01:max(hyperOut);
[surfPlaneVin,surfPlaneVout] = meshgrid(planeVin,planeVout);

% tangent planes
tanPlanePoint1 = [hyperIn(1), hyperOut(1), currentAtHypers(1)];
tanPlanePoint2 = [hyperIn(2), hyperOut(2), currentAtHypers(2)];
tanPlanePoint3 = [hyperIn(3), hyperOut(3), currentAtHypers(3)];
tanPlanePoint4 = [hyperIn(4), hyperOut(4), currentAtHypers(4)];
tanPlaneCurrent1 = zeros(length(planeVout), length(planeVin));
tanPlaneCurrent2 = zeros(length(planeVout), length(planeVin));
tanPlaneCurrent3 = zeros(length(planeVout), length(planeVin));
tanPlaneCurrent4 = zeros(length(planeVout), length(planeVin));

% secant planes
plane1Point1 = [hyperIn(1), hyperOut(1), currentAtHypers(1)];
plane1Point2 = [hyperIn(2), hyperOut(2), currentAtHypers(2)];
plane1Point3 = [hyperIn(3), hyperOut(3), currentAtHypers(3)];
plane1Current = zeros(length(planeVout), length(planeVin));

plane2Point1 = [hyperIn(1), hyperOut(1), currentAtHypers(1)];
plane2Point2 = [hyperIn(2), hyperOut(2), currentAtHypers(2)];
plane2Point3 = [hyperIn(4), hyperOut(4), currentAtHypers(4)];
plane2Current = zeros(length(planeVout), length(planeVin));

plane3Point1 = [hyperIn(2), hyperOut(2), currentAtHypers(2)];
plane3Point2 = [hyperIn(3), hyperOut(3), currentAtHypers(3)];
plane3Point3 = [hyperIn(4), hyperOut(4), currentAtHypers(4)];
plane3Current = zeros(length(planeVout), length(planeVin));

plane4Point1 = [hyperIn(1), hyperOut(1), currentAtHypers(1)];
plane4Point2 = [hyperIn(3), hyperOut(3), currentAtHypers(3)];
plane4Point3 = [hyperIn(4), hyperOut(4), currentAtHypers(4)];
plane4Current = zeros(length(planeVout), length(planeVin));
for i = 1:length(planeVin)
	for j = 1:length(planeVout)
		plane1Current(j,i) = planeEqn(plane1Point1, plane1Point2, plane1Point3, planeVin(i), planeVout(j));
		plane2Current(j,i) = planeEqn(plane2Point1, plane2Point2, plane2Point3, planeVin(i), planeVout(j));
		plane3Current(j,i) = planeEqn(plane3Point1, plane3Point2, plane3Point3, planeVin(i), planeVout(j));
		plane4Current(j,i) = planeEqn(plane4Point1, plane4Point2, plane4Point3, planeVin(i), planeVout(j));
		tanPlaneCurrent1(j,i) = tanEqn(tanPlanePoint1, firDerInAtHypers(1), firDerOutAtHypers(1), planeVin(i), planeVout(j));
		tanPlaneCurrent2(j,i) = tanEqn(tanPlanePoint2, firDerInAtHypers(2), firDerOutAtHypers(2), planeVin(i), planeVout(j));
		tanPlaneCurrent3(j,i) = tanEqn(tanPlanePoint3, firDerInAtHypers(3), firDerOutAtHypers(3), planeVin(i), planeVout(j));
		tanPlaneCurrent4(j,i) = tanEqn(tanPlanePoint4, firDerInAtHypers(4), firDerOutAtHypers(4), planeVin(i), planeVout(j));
	end;
end;

planeCurrent = plane1Current;
tanPlaneCurrent = tanPlaneCurrent4;

%planeCurrent <= I
%tanPlaneCurrent >= I

figure;
s1 = surf(surfVin, surfVout, I, CO);
shading interp;
%hold on;
%plane = surf(surfPlaneVin,surfPlaneVout,planeCurrent);
%hold on;
%tanPlane = surf(surfPlaneVin,surfPlaneVout,tanPlaneCurrent);
%hold on;
%scatter3(hyperIn, hyperOut, currentAtHypers, 'k');
%alpha(plane1, 0.5)
xlabel('Vin')
ylabel('Vout')

%Figure 4 to show convergence around unstable equilibrium for 3 pc
t = 0:0.1:50;
%eps = 0.2;
%y = [0.0030346336306410074 -0.0039304606459618485 0.0034384040635992543 -0.0035117379267088616; 0.004718750563961779 -0.0030346336306410074 0.003664883524410634 -0.006434963356775428]
y = [0.1 0.0 0.1 0.0 0.1 0.0 0.1 0.0];
%perturb = eps*(2*rand(2,4)-1)
%%y = y+perturb
g_cc = 2.0;
%%[t,y] = ode45(@(t,y)oscDot(t,y,a,inverterFunc,g_cc,0),t,y);
[t,y] = ode45(@(t,y)oscDotTransistor(t,y,g_cc),t,y);
figure;
plot(t,y);
xlabel('Time');
ylabel('Voltage');

%currentFun = @currentNFet;
currentFun = @currentPFet;

%src = 0.0;
src = 1.0
gate = 0.5;
drain = 0.0:0.01:1.0;

Vin = drain;

I = zeros(1, length(Vin));
firDerDrain = zeros(1, length(Vin));
secDerDrain = zeros(1, length(Vin))
for i = 1:length(Vin)
	[I(1,i), blah, blah, firDerDrain(1,i), blah, blah, secDerDrain(1,i), blah, blah, blah] = currentFun(src, gate, Vin(i));
end;
secDerDrain;
figure;
plot(Vin, I)
xlabel('Vin')
ylabel('I')

% vary inputVoltages and figure out X_0 for which all node currents are zero
gnd = 0.0;
Vdd = 1.8;
inputVolt = 0.0:0.01:Vdd;
X_0 = 0.0:0.01:Vdd;
X_1 = 0.0:0.01:Vdd;
X_2 = 0.0:0.01:Vdd;

%equilInput = [];
%equilOutput = [];
figure;
%inputVoltages  = 0.0:0.2:Vdd;
inputVoltages = 0.67;
for iv = 1:length(inputVoltages)
	inputVolt = inputVoltages(iv);
	%legendVal = [legendVal, num2str(inputVolt)];
	I_0 = zeros(length(X_0));

	for x0 = 1:length(X_0)
		n_1 = zeros(size(X_1));
		n_2 = zeros(size(X_2));
		for x1 = 1:length(X_1)
			[m_0, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentNFet(gnd, inputVolt, X_1(x1));
			[m_1, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentNFet(X_1(x1), inputVolt, X_0(x0));
			[m_2, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentNFet(X_1(x1), X_0(x0), Vdd);
			n_1(x1) = -m_0 + m_1 + m_2;

		end;
		for x2 = 1:length(X_2)
			[m_3, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentPFet(Vdd, inputVolt, X_0(x0));
			[m_4, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentPFet(X_2(x2), inputVolt, X_0(x0));
			[m_5, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentPFet(X_2(x2), X_0(x0), gnd);
			n_2(x2) = -m_3 + m_4 + m_5;
		end;
		
		zerX1 = NaN;
		zerX2 = NaN;
		zerX1 = bisection(@schmittNfets,gnd, Vdd, inputVolt, X_0(x0), 0.0,Vdd);
		%for i=1:length(n_1)-1
		%	if n_1(i)*n_1(i+1) <= 0.0 
		%		guess1 = X_1(i);
		%		guess2 = X_1(i+1);
		%		zerX1 = bisection(@schmittNfets,gnd, Vdd, inputVolt, X_0(x0), guess1,guess2);
		%		break;
		%	end
		%end;
		if isnan(zerX1)
			disp('didnt find zerX1')
		end;
		%zerX2 = bisection(@schmittPFets,gnd, Vdd, inputVolt, X_0(x0), 0.0,Vdd);
		for i=1:length(n_2)-1
			%if abs(n_2(i)) < 1e-3 | abs(n_2(i+1)) < 1e-3
			%	zerX2 = X_2(i);
			%	break;
			%end;
			if n_2(i)*n_2(i+1) <= 0.0 
				guess1 = X_2(i);
				guess2 = X_2(i+1);
				zerX2 = bisection(@schmittPFets,gnd, Vdd, inputVolt, X_0(x0), guess1,guess2);
				break;
			end
		end;
		if isnan(zerX2)
			n_2
			disp('didnt find zerX2')
		end;
		[m_1, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentNFet(zerX1, inputVolt, X_0(x0));
		[m_4, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentPFet(zerX2, inputVolt, X_0(x0));
		I_0(x0) = -m_1 - m_4;
	end;
	plot(X_0, I_0);
	hold on;
end;

%{
inputVolt = 1.0;
x0 = 1.8002
zerX1 = bisection(@schmittNfets,gnd, Vdd, inputVolt, x0, 0.0, 2.0)
zerX2 = bisection(@schmittPFets,gnd, Vdd, inputVolt, x0, 0.0, 2.0)

x0 = 1.8;
x1 = 0.8001;
x2 = 1.8003;
I1 = schmittNfets(gnd, Vdd, inputVolt, x0, x1)
I2 = schmittPFets(gnd, Vdd, inputVolt, x0, x2)
[m_1, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentNFet(x1, inputVolt, x0);
[m_4, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentPFet(x2, inputVolt, x0);
I0 = -m_1 - m_4
%}


%{
for i = 1:length(inputVolt)
	i
	for x0 = 1:length(X_0)
		%x0
		for x1 = 1:length(X_1)
			for x2 = 1:length(X_2)
				[m_0, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentNFet(gnd, inputVolt(i), X_1(x1));
				[m_1, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentNFet(X_1(x1), inputVolt(i), X_0(x0));
				[m_2, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentNFet(X_1(x1), X_0(x0), Vdd);
				[m_3, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentPFet(Vdd, inputVolt(i), X_0(x0));
				[m_4, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentPFet(X_2(x2), inputVolt(i), X_0(x0));
				[m_5, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentPFet(X_2(x2), X_0(x0), gnd);
				n_0 = -m_4 - m_1;
				n_1 = -m_0 + m_1 + m_2;
				n_2 = -m_3 + m_4 + m_5;
				if abs(n_0) < 1e-8 & abs(n_1) < 1e-8 & abs(n_2) < 1e-8
					equilInput = [equilInput, inputVolt(i)];
					equilOutput = [equilOutput, X_0(x0)];

				end;
			end;
		end;
	end;
end;
%}

%figure;
%plot(equilInput, equilOutput);

%gnd = 0.0;
%Vdd = 1.8;
%inputVoltage = 0.5;
%X_0 = 0.0:0.01:Vdd;
%plotX_0 = [];
%plotI_0 = [];
%for i= 1:length(X_0)
%	[I_0, X_1, X_2] = schmittDebug(inputVoltage, X_0(i), gnd, Vdd);
%	if I_0 ~= NaN
%		plotX_0 = [plotX_0, X_0(i)];
%		plotI_0 = [plotI_0, I_0];
%	end;
%end;
%figure;
%scatter(plotX_0,plotI_0)

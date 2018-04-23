function [I0, X1, X2] = schmittDebug(inputVoltage, X_0, gnd, Vdd)
	volt1 = 0.0:0.01:Vdd;
	volt2 = 0.0:0.01:Vdd;
	X1 = NaN;
	X2 = NaN;
	I0 = NaN;
	n_1 = zeros(size(volt1));
	n_2 = zeros(size(volt2));
	for x1 = 1:length(volt1)
		[m_0, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentNFet(gnd, inputVoltage, volt1(x1));
		[m_1, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentNFet(volt1(x1), inputVoltage, X_0);
		[m_2, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentNFet(volt1(x1), X_0, Vdd);
		n_1(x1) = -m_0 + m_1 + m_2;
		if n_1(x1) == 0
			X1 = volt1(x1);
		end;
	end;

	for x2 = 1:length(volt2)
		[m_3, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentPFet(Vdd, inputVoltage, X_0);
		[m_4, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentPFet(volt2(x2), inputVoltage, X_0);
		[m_5, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentPFet(volt2(x2), X_0, gnd);
		n_2(x2) = -m_3 + m_4 + m_5;
		if n_2(x2) == 0
			X2 = volt2(x2);
		end;
	end;
	if X1 ~= NaN & X2 ~= NaN
		[m1, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentNFet(X1, inputVoltage, X_0);
		[m4, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentPFet(X2, inputVoltage, X_0);
		I0 = -m4 - m1;
	end
	%X_0
	%find(n_1 == 0)
end
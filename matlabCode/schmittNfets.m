function I = schmittNFets(gnd, Vdd, inputVolt, x0, x1)
	[m_0, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentNFet(gnd, inputVolt, x1);
	[m_1, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentNFet(x1, inputVolt, x0);
	[m_2, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentNFet(x1, x0, Vdd);
	%m_0
	%m_1
	%m_2
	I = -m_0 + m_1 + m_2;
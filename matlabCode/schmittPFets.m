function I = schmittPFets(gnd, Vdd, inputVolt, x0, x2)
	[m_3, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentPFet(Vdd, inputVolt, x0);
	[m_4, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentPFet(x2, inputVolt, x0);
	[m_5, blah, blah, blah, blah, blah, blah, blah, blah, blah] = currentPFet(x2, x0, gnd);
	%m_3
	%m_4
	%m_5
	I = -m_3 + m_4 + m_5;
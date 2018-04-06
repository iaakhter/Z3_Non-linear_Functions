/*for(int ii = 0; ii < numVars; ii++){
      lean::numeric_pair<rational> inValSol = m_solver->m_mpq_lar_core_solver.m_r_solver.m_x[inIndices[ii]];
      double inValSolDouble = (inValSol.x + inValSol.y / mpq(1000)).get_double();
      //lean::numeric_pair<rational> outValSol = feasibleSols[outIndices[ii]];
      //std::cout << "outVal " << outValIndices[ii] << " in string format " << T_to_string(feasibleSols[outIndices[ii]]) << "\n";
      //std::cout << "theory var at " << ii << " " << to_app(get_enode(m_var_index2theory_var[inIndices[ii]])->get_owner())->get_decl()->get_name() << "\n";
      //std::cout << "in " << inIndices[ii] <<" in string format " << T_to_string(inValSol) << "\n";
      lean::numeric_pair<rational> minTerm;
      lean::numeric_pair<rational> maxTerm;
      vector<std::pair<rational, lean::var_index> > minCoeffs;
      vector<std::pair<rational, lean::var_index> > maxCoeffs;

      if (m_solver->is_term(inIndices[ii])) {
            const lean::lar_term& term = m_solver->get_term(inIndices[ii]);
            for (auto & ti : term.m_coeffs) {
                  minCoeffs.push_back(std::make_pair(-ti.second, ti.first));
                  maxCoeffs.push_back(std::make_pair(ti.second, ti.first));
            }
      }
      else {
            minCoeffs.push_back(std::make_pair(-rational::one(), inIndices[ii]));
            maxCoeffs.push_back(std::make_pair(rational::one(), inIndices[ii]));
      }
      m_solver->maximize_term(minCoeffs,minTerm);
            //std::cout << "minTerm " << T_to_string(minTerm) << "\n";
            m_solver->get_model(variable_values);
            lean::numeric_pair<rational> updatedVal1 = variable_values[inIndices[ii]];

            m_solver->maximize_term(maxCoeffs,maxTerm);
            //std::cout << "maxTerm " << T_to_string(maxTerm) << "\n";
            m_solver->get_model(variable_values);
            lean::numeric_pair<rational> updatedVal2 = variable_values[inIndices[ii]];

      double val1 = (updatedVal1.x + updatedVal1.y / mpq(1000)).get_double();
      double val2 = (updatedVal2.x + updatedVal2.y / mpq(1000)).get_double();

      hyperRectangle(ii,0) = std::min(val1,val2);
      hyperRectangle(ii,1) = std::max(val1,val2);
}

std::cout << "Check existence within " << hyperRectangle << "\n";
      std::pair<bool,Eigen::MatrixXd> kResult = checkExistenceOfSolution(aVal,params,hyperRectangle,funNum,funDer,funDerInterval);

      double diffNorm = 0.0;
      double inValSolDoubles[numVars];
      double outValSolDoubles[numVars];
      for(int ii = 0; ii<numVars; ii++){
            lean::numeric_pair<rational> inValSol = m_solver->get_value(inIndices[ii]);
            lean::numeric_pair<rational> outValSol = m_solver->get_value(outIndices[ii]);
            inValSolDoubles[ii] = (inValSol.x + inValSol.y / mpq(1000)).get_double();
            outValSolDoubles[ii] = (outValSol.x + outValSol.y/mpq(1000)).get_double();
            diffNorm += (outValSolDoubles[ii] - tanhFun(aVal,inValSolDoubles[ii])) * (outValSolDoubles[ii] - tanhFun(aVal,inValSolDoubles[ii]));
      }
      if(kResult.first){
            hyperRectangle = kResult.second;
      }
      else if (numVars <4 && (diffNorm >= 1e-6 && !kResult.first)){
            constructNewConstraint(aVal,inValSolDoubles, outValSolDoubles,inIndices,outIndices,numVars);
      }
      */

      //if(kResult.first == false && m_stats.m_make_feasible%2 == 0){
      if(false){
            //std::cout << "countAdditionalConstraints " << countAdditionalConstraints << "\n";
            //std::cout << "m_asserted_atoms.size() " << m_asserted_atoms.size() << "\n";
            //std::cout << "CONSTRAINTS after finding hyperrectangle\n";
            //m_solver -> print_constraints(std::cout);
            //std::cout << "\n";
            std::cout << "Voltages " << voltDoubles  << "\n";
            std::cout << "current " << current << "\n";

            std::cout << "REFINING\n";
            std::cout << "Adding Fwd Constraints\n";
            constructNewConstraint(aVal,inValSolDoublesFwd,outValSolDoublesFwd,inIndicesFwd,outIndicesFwd,numVars);
            std::cout << "Adding Cc Constraints\n";
            constructNewConstraint(aVal,inValSolDoublesCc, outValSolDoublesCc, inIndicesCc, outIndicesCc, numVars);
            for(int ii = 0; ii<numVars; ii++){
                  prevVolt(ii,0) = voltDoubles(ii,0);
                  std::vector<double>::iterator itLb;
                  for(int jj = 1; jj<linearBounds.size(); jj++){
                        if(voltDoubles(ii,0) < linearBounds[jj] && voltDoubles(ii,0) > linearBounds[jj-1]){
                              double lowVal = linearBounds[jj-1];
                              double highVal = linearBounds[jj];
                              double valueToEnter = (lowVal+highVal)/2.0;
                              itLb = linearBounds.begin();
                              if(jj == 1 || (jj>=2 && valueToEnter != linearBounds[jj-2])){
                                    linearBounds.insert(itLb+jj,1,valueToEnter);
                              }
                              break;
                        }
                  }
            }
            std::cout << "linearBounds\n";
            for(int ii = 0; ii<linearBounds.size(); ii++){
                  std::cout << linearBounds[ii] << " ";
            }
            std::cout << "\n";
            countAdditionalConstraints++;
      }
}





            /*int sizeOfm_x = m_solver->m_mpq_lar_core_solver.m_r_solver.m_x.size();
            decltype(m_solver->m_mpq_lar_core_solver.m_r_solver.m_x) feasibleSols = m_solver->m_mpq_lar_core_solver.m_r_solver.m_x;
            decltype(m_solver->m_mpq_lar_core_solver.m_r_solver.m_column_types) feasibleColTypes = m_solver->m_mpq_lar_core_solver.m_r_solver.m_column_types;
            for (int tid = 0; tid < sizeOfm_x; tid++){
                  decltype(m_solver->m_mpq_lar_core_solver.m_r_solver.column_name(tid)) feasibleColName = m_solver->m_mpq_lar_core_solver.m_r_solver.column_name(tid);
                  std::string feasColType = "";
                  switch (m_solver->m_mpq_lar_core_solver.m_r_solver.m_column_types[tid]) {
                      case lean::column_type::fixed:
                        feasColType = "fixed";
                        break;
                      case lean::column_type::boxed:
                          feasColType = "boxed";
                          break;
                      case lean::column_type::low_bound:
                          feasColType = "low_bound";
                          break;
                      case lean::column_type::upper_bound:
                          feasColType = "upper_bound";
                          break;
                      case lean::column_type::free_column:
                          feasColType = "free_column";
                          break;
                      default:
                          feasColType = "something else";
                          break;
                      }
                  std::cout << "feasColType " << feasColType << "\n";
                  decltype(feasibleSols[tid]) feasibleSol = feasibleSols[tid];
                  std::cout << "solx at tid " << tid << "\n";
                  feasibleSol.x.display(std::cout);
                  std::cout << "\n";
                  std::cout << "soly at tid " << tid << "\n";
                        feasibleSol.y.display(std::cout);
                        std::cout << "\n";
                  std::cout << "done switch statement\n";
            }*/
            //decltype(feasibleSols[sizeOfm_x-1]) feasibleSol = feasibleSols[sizeOfm_x-1];
            //decltype(m_solver->m_mpq_lar_core_solver.m_r_solver.column_name(sizeOfm_x-1)) feasibleColName = m_solver->m_mpq_lar_core_solver.m_r_solver.column_name(sizeOfm_x-1);
            //vector<unsigned> colToExtVars = m_solver->m_columns_to_ext_vars_or_term_indices;
            //std::unordered_map<unsigned, lean::var_index> extVarToCols = m_solver->getColToExtVar();
            //lean::lar_term* origTerm = m_solver->getOrigTerm()[0];
            //std::cout << "SIZE of m_asserted_atoms " << m_asserted_atoms.size() << "\n";
            //std::cout << "CONSTRAINTS\n";
            //m_solver -> print_constraints(std::cout);
            //std::cout << "\n";
            //m_solver -> print_terms(std::cout);
            //std::cout << "Left side of constraints\n";
            //m_solver->print_left_side_of_constraint(&m_solver->get_constraint(2), std::cout);
            //std::cout << "\n";

                  //std::cout << "inVal displayCol\n";
                  //m_solver->m_mpq_lar_core_solver.m_r_solver.print_column_bound_info(m_theory_var2var_index[inValTheoryVar], std::cout);
                  //std::cout << "outVal displayCol\n";
                  //m_solver->m_mpq_lar_core_solver.m_r_solver.print_column_bound_info(m_theory_var2var_index[outValTheoryVar], std::cout);
                  //lean::column_type inValColType = m_solver->m_mpq_lar_core_solver.m_r_solver.m_column_types[inIndex];
                  //lean::column_type outValColType = m_solver->m_mpq_lar_core_solver.m_r_solver.m_column_types[outIndex];
                  //std::cout << "inValColType " << column_type_to_string(inValColType) << "\n";
                  //std::cout << "outValColType " << column_type_to_string(outValColType) << "\n";




distance = 0.2;
Eigen::MatrixXd oneMatrix(numVars,1);
oneMatrix = Eigen::MatrixXd::Ones(numVars,1)*distance;
std::cout << "voltDoubles " << voltDoubles << "\n";
hyperRectangle.col(0) = voltDoubles - oneMatrix;
hyperRectangle.col(1) = voltDoubles + oneMatrix;
Eigen::MatrixXd oldHyper(numVars,2);
oldHyper.col(0) = voltDoubles - oneMatrix;
oldHyper.col(1) = voltDoubles + oneMatrix;
std::pair<bool,Eigen::MatrixXd> firstKResult = checkExistenceOfSolution(aVal,params,hyperRectangle,funNum,funDer,funDerInterval);
bool discardHyperRectangle = false;
bool addedToHyperList = false;
while(!discardHyperRectangle){
  std::pair<bool,Eigen::MatrixXd> kResult = checkExistenceOfSolution(aVal,params,hyperRectangle,funNum,funDer,funDerInterval);
  if (kResult.first==true || (kResult.first == false && isinf(kResult.second(0,0)))){
    if(kResult.first){
      //Eigen::MatrixXd hyperCopy(numVars,2);
      //hyperCopy = kResult.second.replicate(numVars,2);
      if(allUniqueHyperRectangles.size() == 0 ||
          allUniqueHyperRectangles.back() != kResult.second){
        allUniqueHyperRectangles.push_back(kResult.second);
        addedToHyperList = true;
      }
      hyperRectangle = kResult.second;
    }
    discardHyperRectangle = true;
    if(addedToHyperList || !kResult.first){
      ignoreCurrentSolution(hyperRectangle,inIndices);
    }
  }
  else if (kResult.first == false && !isinf(kResult.second(0,0))){
    distance = distance/2.0;
    oneMatrix = Eigen::MatrixXd::Ones(numVars,1)*distance;
    hyperRectangle.col(0) = voltDoubles - oneMatrix;
    hyperRectangle.col(1) = voltDoubles + oneMatrix;
  }
}
//Eigen::MatrixXd
if(firstKResult.first == false){
//if(true){
  //if(!isinf(firstKResult.second(0,0))){
  //  hyperRectangle = firstKResult.second;
  //}

  Eigen::MatrixXd hyper1(numVars,2);
  hyper1 = Eigen::MatrixXd::Constant(numVars,2,std::numeric_limits<double>::infinity());
  Eigen::MatrixXd hyper2(numVars,2);
  hyper2 = Eigen::MatrixXd::Constant(numVars,2,std::numeric_limits<double>::infinity());
  /*hyper1.col(0) = oldHyper.col(0);
  hyper1.col(1) = voltDoubles;
  hyper2.col(0) = voltDoubles;
  hyper2.col(1) = oldHyper.col(1);*/
  bool properInsertion = true;
  for(int ii = 0; ii<numVars; ii++){
    if(hyperRectangle(ii,0) < 0 && hyperRectangle(ii,1) > 0){
      if(voltDoubles(ii,0) < 0){
        hyperRectangle(ii,1) = 0.0;
      }
      else{
        hyperRectangle(ii,0) = 0.0;
      }
    }
    /*std::vector<double> lbVector = linearBounds[ii];
    std::vector<double>::iterator it = lbVector.begin();
    std::cout << "voltDouble at " << ii <<": "<< voltDoubles(ii,0) << "\n";
    int oldSize = lbVector.size();
    for(int lbi = 1; lbi < lbVector.size(); lbi++){
      if(voltDoubles(ii,0) > lbVector[lbi-1] && voltDoubles(ii,0) < lbVector[lbi]){
        //std::cout << "Inserted between " << lbVector[lbi-1] << " and " << lbVector[lbi] << "\n";
        hyper1(ii,0) = lbVector[lbi-1];
        hyper1(ii,1) = voltDoubles(ii,0);
        hyper2(ii,0) = voltDoubles(ii,0);
        hyper2(ii,1) = lbVector[lbi];
        lbVector.insert(it+lbi,1,voltDoubles(ii,0));
        break;
      }
    }
    if (lbVector.size() <= oldSize)
      properInsertion = false;
    linearBounds[ii] = lbVector;
    std::cout << "linearBounds for var "<< ii <<" contains:";
    for (it=linearBounds[ii].begin(); it<linearBounds[ii].end(); it++)
      std::cout << ' ' << *it;
    std::cout << '\n';*/
  }

  std::cout << "Adding triangle constraint\n";
  //std::cout << "Within " << hyperRectangle << "\n";
  /*std::cout << "hyper1 " << hyper1 << "\n";
  std::cout << "hyper2 " << hyper2 << "\n";
  Eigen::MatrixXd hyper1Col = hyper1.col(1) - hyper1.col(0);
  Eigen::MatrixXd hyper2Col = hyper2.col(1) - hyper2.col(0);

  if (properInsertion){
    addTriangleConstraints(aVal, inIndicesFwd, outIndicesFwd, voltIndicesFwd, hyper1);
    addTriangleConstraints(aVal, inIndicesCc, outIndicesCc, voltIndicesCc, hyper1);
  }
  if (properInsertion){
    addTriangleConstraints(aVal, inIndicesFwd, outIndicesFwd, voltIndicesFwd, hyper2);
    addTriangleConstraints(aVal, inIndicesCc, outIndicesCc, voltIndicesCc, hyper2);
  }*/
  addTriangleConstraints(aVal, inIndicesFwd, outIndicesFwd, voltIndicesFwd, hyperRectangle);
  addTriangleConstraints(aVal, inIndicesCc, outIndicesCc, voltIndicesCc, hyperRectangle);


}

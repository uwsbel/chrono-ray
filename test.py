from ChR_Config import ChR_Distr

d = ChR_Distr.grid_search([0.01, 0.1, 1.0])
print(ChR_Distr._format_distr(d))
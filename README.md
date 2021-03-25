# Portfolio-DataScience--Estudos
 Coletânia de códigos desenvolvidos em cursos e bootcamps

# Machine Learning e Data Science com Python
Seção 3 - Pré-processamento com pandas e scikit-learn (Base: credit_data)

	Bibliotecas:
		- pandas
		- scikit-learn (Imputer e StandardScaler)

	funções:
		pd.read_csv()	-- ler arquivo csv
		describe()	-- estatística básica df
		loc[]		-- localização de valores
		drop()		-- apagar dados do df
		mean()		-- mostra a média do atributo
		pd.isnull()	-- compara os valores de um atributo e marca verdadeiro se for nulo
		iloc[]		-- localização de valores
		fit()		-- calcula a média das colunas de dados (Regra para Imputer)
		transform()	-- substitui valores ausentes
		fit_transform()	-- calcula a padronização e substitui todos os valores pelo resultado (Regra para StandardScaler)

	Métodos de tratamentos de dados inconsistentes:
		- apagar a coluna
		- apagar os registros
		- preencher os valores com a média

	Tratamentos de dados faltantes
		- Localizar dados faltantes
		- Separar base para construção de ML
		- Tratar os dados com Imputer

	Escalonamento de atributos
		- Padronização (Standardisation)	-- implementação
		- Normalização (Normalization)

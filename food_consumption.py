from pca import PCA
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# NOTE: Do not change the parameters / return types for pre defined methods.
# -----------------------------------------------------------------------------
class FoodConsumptionPCA:
    """
    This class loads and processes a food consumption dataset for performing PCA
    from two perspectives:
    - Country-based PCA: Countries as samples, foods as features.
    - Food-based PCA: Foods as samples, country consumption patterns as features.
    """

    def __init__(self, input_path="data/food-consumption.csv"):
        """
        Initializes the FoodConsumptionPCA object and loads data from a CSV file.

        Parameters:
        ----------
        input_path : str
            Path to the CSV file containing the food consumption data.
        """

        ##Read the CSV file
        with open(input_path, 'r') as file:
            lines = file.read().strip().split('\n')
        
        ##Remove the headers
        headers = lines[0].split(',')

        ##Remove the country names
        self.foods = headers[1:]

        ##Get the rows
        data_rows = []
        countries = []
        ##Skip the header
        for line in lines[1:]:
            ##Save the countries
            indv_line = line.split(',')
            countries.append(indv_line[0])
            ##Save row data and convert to float based on troubleshooting
            row_data = [float(x) for x in indv_line[1:]]
            data_rows.append(row_data)

        ##Convert this to numpy arrays
        self.countries = np.array(countries)
        self.food_data = np.array(data_rows)
        
    def country_pca(self, num_dim: int) -> np.ndarray:
        """
        Performs PCA where each row represents a country and each column represents a food item.

        This will reduce the feature space (foods) to `num_dim` principal components.

        Parameters:
        ----------
        num_dim : int
            Number of principal components to retain.

        Returns:
        -------
        np.ndarray
            A (num_countries, num_dim) array representing countries in the reduced PCA space.
        """
        
        ##Call the PCA function to initialize
        pca = PCA()

        ##Transform the data and run PCA
        final = pca.fit_transform(self.food_data, num_dim)
        return final

    def food_pca(self, num_dim: int) -> np.ndarray:
        """
        Performs PCA where each row represents a food item and each column represents a country.

        This will reduce the country-dimension feature space to `num_dim` principal components.

        Parameters:
        ----------
        num_dim : int
            Number of principal components to retain.

        Returns:
        -------
        np.ndarray
            A (num_foods, num_dim) array representing foods in the reduced PCA space.
        """
        
        ##Call the PCA function to initialize
        pca = PCA()

        ##Transpose the foot_data because the columns and rows are flipped
        food_data_T = self.food_data.T

        ##Similar as before, return the array that shows foods with the smaller dimensions
        final = pca.fit_transform(food_data_T, num_dim)
        return final
    
    def run_country(self):
        ##Run and plot everything

        ##Number of dimensions is 2
        num_dim = 2

        ##Preform country based PCA
        country_pca_result = self.country_pca(num_dim)

        ##Create the plot 
        plt.figure(figsize = (10,10))
        plt.scatter(country_pca_result[:,0], country_pca_result[:,1], alpha = 0.7, s = 100)

        ##Cycle thru each point and label the country
        for i,country in enumerate(self.countries):
            plt.annotate(country, (country_pca_result[i,0],country_pca_result[i,1]), xytext = (5,5), textcoords = 'offset points', fontsize = 8)
        
        plt.title('Country-Based PCA')
        plt.grid(True, alpha = 0.3)
        plt.show()

    def run_food(self):
        ##Run and plot everything

        ##Number of dimensions is 2
        num_dim = 2

        ##Preform country based PCA
        food_pca_result = self.food_pca(num_dim)

        ##Create the plot 
        plt.figure(figsize = (10,10))
        plt.scatter(food_pca_result[:,0], food_pca_result[:,1], alpha = 0.7, s = 100)

        ##Cycle thru each point and label the country
        for i,food in enumerate(self.foods):
            plt.annotate(food, (food_pca_result[i,0],food_pca_result[i,1]), xytext = (5,5), textcoords = 'offset points', fontsize = 8)
        
        plt.title('Food-Based PCA')
        plt.grid(True, alpha = 0.3)
        plt.show()

country_food = FoodConsumptionPCA()
country_food.run_country()
country_food.run_food()
        


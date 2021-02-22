/*
 ============================================================================
 Name        : EEGClassificationWithANN.c
 Author      : Sushant Kumar & Abdulvahap Calikoglu
 Version     : Version 1.1
 Copyright   : Copyright © 2020 Bremen, Germany. All rights are reserved.
 Description : Artificial neural network implementation in C for EEG Classification.
 ============================================================================
 */

// Import libraries
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


// Define variables to calculate duration
clock_t start_time, end_time;
double time_used;

// Define size of the layers
int neurons_in_input_layer = 178;
int neurons_in_hidden_layer1 = 32;
int neurons_in_hidden_layer2 = 32;
int neurons_in_output_layer = 1;

// Define arrays to store biases
float bias_for_layer1[32];
float bias_for_layer2[32];
float bias_for_layer3[1];

// Define arrays to store data and weights
float test_data_inputs[2300][178];
float test_data_outputs[2300][1];
float weights_for_layer1[178][32];
float weights_for_layer2[32][32];
float weights_for_output_layer[32][1];

// Define function prototypes to read and store data, network parameters and to calculate model output
void load_bias_for_layer1(FILE *file_path);
void load_bias_for_layer2(FILE *file_path);
void load_bias_for_layer3(FILE *file_path);
void read_input_file(FILE *file_path);
void read_label_file(FILE *file_path);
void load_weights_for_layer1(FILE *file_path);
void load_weights_for_layer_2(FILE *file_path);
void load_weights_for_output_layer(FILE *file_path);
int* calculate_model_output();

int main(void) {

	// Read the input file which needs to be tested
	read_input_file("src/in.csv");

	// Part 1.2: Reading Output labels File, which needs to be compare for accuracy testing
	read_label_file("src/out.csv");

	// Read weights for hidden layers 1
	load_weights_for_layer1("src/one.csv");

	// Read weights for hidden layers 2
	load_weights_for_layer_2("src/two.csv");

	// Read weights for output layer
	load_weights_for_output_layer("src/three.csv");

	// Read biases for layer 1
	load_bias_for_layer1("src/bias1.csv");

	// Read biases for layer 2
	load_bias_for_layer2("src/bias2.csv");

	// Read biases for output layer
	load_bias_for_layer3("src/bias3.csv");

	// Count correct and incorrect predictions to calculate model accuracy
	double correct_predictions = 0, incorrect_predictions = 0;

	// Save the current time to calculate classification duration
	start_time = clock();

	int *model_output_pointer;
	model_output_pointer = calculate_model_output(); //Pointer to array

	// Save the current time to calculate classification duration
	end_time = clock();

	// Count correct and incorrect predictions to calculate accuracy
	int l = 0;
	for (l = 0; l < 2300; l++) {
		if (*(model_output_pointer + l) == test_data_outputs[l][0]) {
			correct_predictions = correct_predictions + 1;
		} else {
			incorrect_predictions = incorrect_predictions + 1;
		}
	}

	int samples = 2300;

	// Print model accuracy and classification duration for 2300 samples
	double model_accuracy = (correct_predictions
			/ (correct_predictions + incorrect_predictions)) * 100;
	printf("Accuracy of model for %d samples = %f percent \n\r", samples,
			model_accuracy);

	time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
	printf("Testing time for %d samples = %f seconds \n", samples, time_used);
}


/*********************************************************************************
 =========================READING INPUT FILES======================================
 *********************************************************************************/

void read_input_file(FILE *file_path) {
	// Define arrays and pointers to store and extract data
	char temporary_storage[2500];
	int number_of_samples_to_be_read = 2500;
	char *time_points;
	char *sample_values;
	int data_row1 = 0, data_column1 = 0;

	FILE *input_file_pointer;
	// Open file and check if it is opened successfully
	if ((input_file_pointer = fopen(file_path, "r")) != NULL) {

		// Get all the lines (recordings) in the file to extract until there is not any line left
		while ((time_points = fgets(temporary_storage,
				number_of_samples_to_be_read, input_file_pointer)) != NULL)
		{
			// Define a separator to extract each value in one recording
			const char separator[] = ",";
			// Extract first value
			sample_values = strtok(time_points, separator);
			// Keep looping until there is not any value left in each recording
			while (sample_values != NULL) {
				// Store float value of the value in the array to use in the code
				test_data_inputs[data_row1][data_column1++] = atof(
						sample_values);
				// Print value to see
				// puts(sample_values);
				// Go out of the first loop when there is no any value left and continue with next row (recording)
				sample_values = strtok(NULL, separator);
			}
			// Reset index of the value in a recording for next recording extraction
			data_column1 = 0;
			// Increment index of recording for next recording extraction
			data_row1++;
		}
	} else {
		// Inform user when file is not opened properly
		printf("Input file is not available in the project folder ! \n");

	}

}

/*********************************************************************************
 =========================READING LABELS FILES=====================================
 *********************************************************************************/

void read_label_file(FILE *file_path) {
	// Define arrays and pointers to store and extract data
	char temporary_storage[2500];
	int number_of_samples_to_be_read = 2500;
	char *time_points;
	char *sample_labels;
	int data_row2 = 0, data_column2 = 0;

	FILE *output_file_pointer;
	// Open file and check if it is opened successfully
	if ((output_file_pointer = fopen(file_path, "r")) != NULL) {

		// Get all the lines (labels) in the file to extract until there is not any line left
		while ((time_points = fgets(temporary_storage,
				number_of_samples_to_be_read, output_file_pointer)) != NULL)
		{
			// Define a separator to extract each label
			const char separator[] = ",";
			// Extract first label
			sample_labels = strtok(time_points, separator);
			// Keep looping until there is not any label left
			while (sample_labels != NULL) {
				// Store float value of the label in the array to use in the code
				test_data_outputs[data_row2][data_column2++] = atof(
						sample_labels);
				// Print label to see
				// puts(sample_labels);
				// Go out of the first loop when there is no any label left and continue with next row (recording's label)
				sample_labels = strtok(NULL, separator);
			}
			// Reset index of the label for next label extraction
			data_column2 = 0;
			// Increment index of recording's label for next recording's label extraction
			data_row2++;
		}

	} else {
		// Inform user when file is not opened properly
		printf("Output label file is not available in the project folder ! \n");

	}
}

/*********************************************************************************
 =========================READING WEIGHTS of LAYER 1 FILES=========================
 *********************************************************************************/

void load_weights_for_layer1(FILE *file_path) {
	// Define arrays and pointers to store and extract data
	char temporary_storage[2500];
	int number_of_samples_to_be_read = 2500;
	char *line1 = NULL;
	char *value1 = NULL;
	int weight_row1 = 0, weight_column1 = 0;

	FILE *weights_for_layer1_pointer;
	// Open file and check if it is opened successfully
	if ((weights_for_layer1_pointer = fopen(file_path, "r")) != NULL) {

		// Get all the lines (weights) in the file to extract until there is not any weight line left
		while ((line1 = fgets(temporary_storage, number_of_samples_to_be_read,
				weights_for_layer1_pointer)) != NULL) {
			// Define a separator to extract each weight
			const char separator[] = ",";
			// Extract first weight
			value1 = strtok(line1, separator);
			// Keep looping until there is not any weight left
			while (value1 != NULL) {
				// Store float value of the weight in the array to use in the code
				weights_for_layer1[weight_row1][weight_column1++] = atof(
						value1);
				// Print weight to see
				// puts(value1);
				// Go out of the first loop when there is no any weight left and continue with next row
				value1 = strtok(NULL, separator);
			}
			// Reset index of the weight for next weight sample extraction
			weight_column1 = 0;
			// Increment index of weight for next weight extraction in a row
			weight_row1++;
		}

	} else {
		// Inform user when file is not opened properly
		printf("Weights for layer 1 is not available in the project folder ! \n");
	}
}

/*********************************************************************************
 =========================READING WEIGHTS of LAYER 2 FILES=========================
 *********************************************************************************/

void load_weights_for_layer_2(FILE *file_path) {
	char temporary_storage[2500];
	int number_of_samples_to_be_read = 2500;
	char *line2 = NULL;
	char *value2 = NULL;
	int weight_row2 = 0, weight_column2 = 0;

	FILE *weights_for_layer2_pointer;
	if ((weights_for_layer2_pointer = fopen(file_path, "r")) != NULL) {

		while ((line2 = fgets(temporary_storage, number_of_samples_to_be_read,
				weights_for_layer2_pointer)) != NULL) {
			const char s[] = ",";
			value2 = strtok(line2, s);
			while (value2 != NULL) {
				weights_for_layer2[weight_row2][weight_column2++] = atof(
						value2);
				//puts(value2);

				value2 = strtok(NULL, s);
			}
			weight_column2 = 0;
			weight_row2++;
		}

	} else {
		printf(
				"Weights for layer 2 is not available in the project folder ! \n");
	}
}

/*********************************************************************************
 =========================READING WEIGHTS of OUTPUT LAYER FILES====================
 *********************************************************************************/

void load_weights_for_output_layer(FILE *file_path) {
	char temporaray_storage[2500];
	int number_of_samples_to_be_read = 2500;
	char *line3 = NULL;
	char *value3 = NULL;
	int weight_row3 = 0, weight_column3 = 0;

	FILE *weights_for_output_layer_pointer;
	if ((weights_for_output_layer_pointer = fopen(file_path, "r")) != NULL) {

		while ((line3 = fgets(temporaray_storage, number_of_samples_to_be_read,
				weights_for_output_layer_pointer)) != NULL) {
			const char s[] = ",";
			value3 = strtok(line3, s);
			while (value3 != NULL) {
				weights_for_output_layer[weight_row3][weight_column3++] = atof(
						value3);
				//puts(value3);

				value3 = strtok(NULL, s);
			}
			weight_column3 = 0;
			weight_row3++;
		}
	} else {
		printf(
				"Weights for output layer is not available in the project folder ! \n");

	}

}

/*********************************************************************************
 =========================READING BIASES of FIRST HIDDEN LAYER ===================
 *********************************************************************************/


void load_bias_for_layer1(FILE *file_path) {
	// Define arrays and pointers to store and extract data
	char temporary_storage[2500];
	int number_of_samples_to_be_read = 2500;
	char *bias_line1 = NULL;
	char *bias_value1 = NULL;
	int bias_row1 = 0, bias_column1 = 0;

	FILE *weights_for_layer1_pointer;
	// Open file and check if it is opened successfully
	if ((weights_for_layer1_pointer = fopen(file_path, "r")) != NULL) {
		// Get all the lines (biases) in the file to extract until there is not any bias line left
		while ((bias_line1 = fgets(temporary_storage, number_of_samples_to_be_read,
				weights_for_layer1_pointer)) != NULL) {
			// Define a separator to extract each bias
			const char separator[] = ",";
			// Extract first bias
			bias_value1 = strtok(bias_line1, separator);
			// Keep looping until there is not any bias left
			while (bias_value1 != NULL) {
				// Store float value of the bias in the array to use in the code
				bias_for_layer1[bias_row1] = atof(
						bias_value1);
				// Print bias to see
				// puts(bias_value1);
				// Go out of the first loop when there is no any bias left and continue with next row
				bias_value1 = strtok(NULL, separator);
			}
			// Increment the index of bias for the next neuron
			bias_row1++;
		}

	} else {
		printf("Biases for layer 1 is not available in the project folder ! \n");
	}


}


/*********************************************************************************
 =========================READING BIASES of SECOND HIDDEN LAYER ==================
 *********************************************************************************/

void load_bias_for_layer2(FILE *file_path) {


	char temporary_storage[2500];
	int number_of_samples_to_be_read = 2500;
	char *bias_line2 = NULL;
	char *bias_value2 = NULL;
	int bias_row2 = 0, bias_column2 = 0;

	FILE *bias_for_layer2_pointer;
	if ((bias_for_layer2_pointer = fopen(file_path, "r")) != NULL) {

		while ((bias_line2 = fgets(temporary_storage, number_of_samples_to_be_read,
				bias_for_layer2_pointer)) != NULL) {
			const char separator[] = ",";
			bias_value2 = strtok(bias_line2, separator);
			while (bias_value2 != NULL) {
				bias_for_layer2[bias_row2] = atof(
						bias_value2);
				// puts(bias_value1);

				bias_value2 = strtok(NULL, separator);
			}
			bias_row2++;
		}

	} else {
		printf("Biases for layer 2 is not available in the project folder ! \n");
	}


}

/*********************************************************************************
 =========================READING BIAS of OUTPUT LAYER ===========================
 *********************************************************************************/

void load_bias_for_layer3(FILE *file_path) {

		char temporary_storage[2500];
		int number_of_samples_to_be_read = 2500;
		char *bias_line3 = NULL;
		char *bias_value3 = NULL;
		int bias_row3 = 0, bias_cloumn3 = 0;

		FILE *bias_for_layer3_pointer;
		if ((bias_for_layer3_pointer = fopen(file_path, "r")) != NULL) {

			while ((bias_line3 = fgets(temporary_storage, number_of_samples_to_be_read,
					bias_for_layer3_pointer)) != NULL) {
				const char separator[] = ",";
				bias_value3 = strtok(bias_line3, separator);
				while (bias_value3 != NULL) {
					bias_for_layer2[bias_row3] = atof(
							bias_value3);
					// puts(bias_value1);

					bias_value3 = strtok(NULL, separator);
				}
				bias_row3++;
			}

		} else {
			printf("Biases for layer 3 is not available in the project folder ! \n");
		}

}



/*********************************************************************************
 ==========================CALCULATIONS of MODEL OUTPUT============================
 *********************************************************************************/

int* calculate_model_output() {

	float hidden_layer1[neurons_in_hidden_layer1];
	float hidden_layer2[neurons_in_hidden_layer2];
	int i = 0, j = 0;
	int test_inputs = 0;
	float output[2300];
	static int model_out[2300];

	for (test_inputs = 0; test_inputs < 2300; test_inputs++) {

		// First hidden layer calculations

		float layer_1_product = 0, layer_1_sum = 0;
		for (i = 0; i < neurons_in_hidden_layer1; i++) {
			for (j = 0; j < neurons_in_input_layer; j++) {
				layer_1_product = test_data_inputs[test_inputs][j]
						* weights_for_layer1[j][i]; // X1 * W1-1
				layer_1_sum = layer_1_sum + layer_1_product;
			}


			layer_1_sum = layer_1_sum + bias_for_layer1[i];


			//RELU function implementation
			if (layer_1_sum < 0) {
				hidden_layer1[i] = 0;
			} else {
				hidden_layer1[i] = layer_1_sum;
			}

			layer_1_sum = 0;
			layer_1_product = 0;
		}

		// Second hidden layer calculations
		float layer_2_product = 0, layer_2_sum = 0;

		for (i = 0; i < neurons_in_hidden_layer2; i++) {
			for (j = 0; j < neurons_in_hidden_layer1; j++) {
				layer_2_product = hidden_layer1[j] * weights_for_layer2[j][i];// N1 * W2-1
				layer_2_sum = layer_2_sum + layer_2_product;
			}


			layer_2_sum = layer_2_sum + bias_for_layer2[i];


			//RELU function implementation
			if (layer_2_sum < 0) {
				hidden_layer2[i] = 0;
			} else {
				hidden_layer2[i] = layer_2_sum;
			}
			layer_2_sum = 0;
			layer_2_product = 0;
		}


		// Output Layer Calculations

		float output_layer_product = 0, output_layer_sum = 0;

		for (i = 0; i < neurons_in_hidden_layer2; i++) {
			output_layer_product = hidden_layer2[i]
					* weights_for_output_layer[i][0];
			output_layer_sum = output_layer_sum + output_layer_product;
		}

		output_layer_sum = output_layer_sum + bias_for_layer3[0];

		// SIGMOID function implementation
		output[test_inputs] =
				(1.0 / (1.0 + exp((float) (-(output_layer_sum)))));

		output_layer_product = 0;
		output_layer_sum = 0;

		if (output[test_inputs] < 0.5) {
			model_out[test_inputs] = 0;
		} else {
			model_out[test_inputs] = 1;
		}

	}

	return model_out;
}


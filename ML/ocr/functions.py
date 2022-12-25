import numpy as np

__verify_soma_msg_error = "A variável soma deve ser numérica."
__verify_softmax_msg_error = "A variável vetor_sum deve ser uma lista."
__verify_accuracy_lists_len_msg_error = '(calculated, actual) devem ter os mesmos tamanhos'
__verify_accuracy_args_type_msg_error = '(calculated, actual) devem ser listas.'
__verify_accuracy_lists_values_msg_error = 'Os valores contidos nas listas(calculated, actual) devem ser numéricos.'


def __verify_soma(val: any) -> None:
    """
    Função apenas para avaliar se o parametro é condizente com o formato esperado.
    :param val: valor a ser avaliado.
    :return: None: Apenas realiza o rise se os valores estiverem fora do esperado.
    """
    if not isinstance(val, (int, float)):
        raise TypeError(__verify_soma_msg_error)


def __verify_accuracy_args_type(calculated, actual):
    if not isinstance(calculated, list) or not isinstance(actual, list):
        raise TypeError(__verify_accuracy_args_type_msg_error)


def __verify_accuracy_lists_len(calculated, actual):
    if not len(actual) == len(calculated):
        raise ValueError(__verify_accuracy_lists_len_msg_error)


def __verify_accuracy_lists_values(calculated: list, actual: list):
    if True in [not isinstance(i, (int, float)) for i in calculated] or \
            True in [not isinstance(i, (int, float)) for i in actual]:
        raise TypeError(__verify_accuracy_lists_values_msg_error)


def __verify_softmax_arg_type(to_test: list) -> None:
    if not isinstance(to_test, list):
        raise TypeError(__verify_softmax_msg_error)

def step_function(soma: float) -> int:
    """
    Função de classificação linear do modelo Step Function, com retornos equivalentes as 0 ou 1
    :param soma: a soma resultante do(s) perseptron(s) a ser avaliada pela function
    :return: int: 0 ou 1 conforme avaliação da soma.
    """
    __verify_soma(soma)
    if soma >= 1:
        return 1
    return 0


def sigmoid_function(soma: float) -> float:
    """
    Função não linear, para classificação em resultados XOR
    :param soma: a soma resultante do(s) perseptron(s) a ser avaliada pela function
    :return: float: entre 0 e 1 conforme avaliação da soma.
    """
    __verify_soma(soma)
    return 1 / (1 + np.exp(-soma))


def tahn_function(soma: float) -> float:
    """
    Função não linear, para classificação em resultados XOR
    :param soma: a soma resultante do(s) perseptron(s) a ser avaliada pela function
    :return: float: entre 0 e 1 conforme avaliação da soma.
    """
    __verify_soma(soma)
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))


def relu_function(soma: float) -> float:
    """
    RElu Function, função que realiza o crop de valores negativos.
    :param soma: a soma resultante do(s) perseptron(s) a ser avaliada pela function
    :return:
    """
    __verify_soma(soma)
    return soma if soma >= 0 else 0


def linear_function(soma: float) -> float:
    """
    Esta função apenas retorna o valor passado.
    :param soma: valor de soma.
    :return: soma
    """
    __verify_soma(soma)
    return soma


def softmax_function(vetor_sum: list):
    __verify_softmax_arg_type(vetor_sum)
    ex = np.exp(vetor_sum)
    return ex / ex.sum()


# usar o Keras para mais functions

def accuracy_tx_by_mae(calculated: list, actual: list) -> float:
    # todo montar testes e docmuentar
    __verify_accuracy_args_type(calculated, actual)
    __verify_accuracy_lists_len(calculated, actual)
    __verify_accuracy_lists_values(calculated, actual)
    n = len(calculated)
    my_sum = 0
    for i in range(n):
        my_sum += abs(actual[i] - calculated[i])
    return my_sum / n


def accuracy_tx_by_mse(calculated: list, actual: list) -> float:
    # todo montar testes e documentar
    __verify_accuracy_lists_len(calculated, actual)
    __verify_accuracy_lists_values(calculated, actual)
    __verify_accuracy_args_type(calculated, actual)
    return np.square(np.subtract(calculated, actual)).mean()




if __name__ == '__main__':
    i = accuracy_tx_by_mae([1, 0, 1, 0], [0.3, 0.02, 0.89, 0.32])
    print(i)
    i = accuracy_tx_by_mse([1, 0, 1, 0], [0.3, 0.02, 0.89, 0.32])
    print(i)

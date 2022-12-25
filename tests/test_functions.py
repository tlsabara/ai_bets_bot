import pytest
import ML.ocr.functions as functions


# RAISES VALIDATION
def test_verfy_function_shold_return_none():
    """
    Verificar se a função esta retornando None para int e float.
    """
    assert functions.__verify_soma(1) is None
    assert functions.__verify_soma(1.999) is None


@pytest.mark.parametrize(
    'value',
    [
        ('str',),
        (True,),
        (['ss'],)
    ]
)
def test_verfy_function_shold_raise_error_if_non_numerical_parameter(value):
    """
    Verificar se a função esta fazendo rise corretamente.
    """
    with pytest.raises(TypeError) as e:
        functions.__verify_soma(value)


@pytest.mark.parametrize(
    'func, soma',
    [
        (functions.step_function, 'any_string'),
        (functions.tahn_function, 'any_string'),
        (functions.sigmoid_function, 'any_string'),
        (functions.relu_function, 'any_string'),
        (functions.linear_function, 'any_string'),
    ]
)
def test_step_function_shold_raise_error_if_recive_non_numerical_parameters(func, soma):
    """
    Para testar se a função esta aceitando apenas numeros.
    """
    error = None
    with pytest.raises(TypeError):
        try:
            func(soma)
        except Exception as e:
            error = e.args
            raise e
    assert len(error) == 1
    assert error[0] == functions.__verify_soma_message_error


# STEP FUNCTION.
@pytest.mark.parametrize(
    'soma, expected',
    [
        (-10, 0),
        (-1, 0),
        (1, 1),
        (10, 1)
    ]
)
def test_step_function_shold_return_expected_values(soma, expected):
    """
    Teste para validar o funcionamento esperado, 0 para valores menores que 0, 1 para valores maiores)
    """
    assert functions.step_function(soma=soma) == expected


# SIGMOID FUNCTION
@pytest.mark.parametrize(
    'soma, expected',
    [
        (1, 0.7310585786300049),
        (10, 0.9999546021312976),
        (-1, 0.2689414213699951),
        (-10, 4.5397868702434395e-05),
    ]
)
def test_sigmoid_function_shold_return_expected_values(soma, expected):
    """
    Teste para validar o funcionamento esperado, 0 para valores menores que 0, 1 para valores maiores)
    """
    assert functions.sigmoid_function(soma) == expected


# TAHN FUNCTION
@pytest.mark.parametrize(
    'soma, expected',
    [
        (1, 0.7615941559557649),
        (10, 0.9999999958776926),
        (-1, -0.7615941559557649),
        (-10, -0.9999999958776926),
    ]
)
def test_tahn_function_shold_return_expected_values(soma, expected):
    """
    Teste para validar o funcionamento esperado, 0 para valores menores que 0, 1 para valores maiores)
    """
    assert functions.tahn_function(soma) == expected


# RELU FUNCTION
@pytest.mark.parametrize(
    'soma, expected',
    [
        (1, 1),
        (10, 10),
        (-1, 0),
        (-10, 0),
    ]
)
def test_relu_function_shold_return_expected_values(soma, expected):
    """
    Teste para validar o funcionamento esperado, 0 para valores menores que 0, 1 para valores maiores)
    """
    assert functions.relu_function(soma) == expected


# LINEAR FUNCTION
@pytest.mark.parametrize(
    'soma, expected',
    [
        (1, 1),
        (10, 10),
        (-1, -1),
        (55, 55),
    ]
)
def test_linear_function_shold_return_expected_values(soma, expected):
    """
    Teste para validar o funcionamento esperado, 0 para valores menores que 0, 1 para valores maiores)
    """
    assert functions.linear_function(soma) == expected


# SOFTMAX FUNCTION
@pytest.mark.parametrize(
    'soma, expected',
    [
        ([1, 1, 1, 1], [0.25, 0.25, 0.25, 0.25]),
        ([10, 10, 10, 10], [0.25, 0.25, 0.25, 0.25]),
        ([4, 3, 2, 1], [0.6439142598879722, 0.23688281808991013, 0.08714431874203257, 0.03205860328008499]),
        ([0.543, 0.441, 5.34, 3], [0.007422846279256795, 0.006703049547664254, 0.8992513458516018, 0.08662275832147708])
    ]
)
def test_softmax_function_shold_return_expected_values(soma, expected):
    """
    Teste para validar o funcionamento esperado, 0 para valores menores que 0, 1 para valores maiores)
    """
    assert [i for i in functions.softmax_function(soma)] == expected

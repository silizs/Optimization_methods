import scipy as sp
import numpy as np


# Функция возвращает константый заданный шаг
def constant_step(step_size, x_last, function, gradient, eps):
    return np.array([step_size, 0])


# Функция возвращает шаг, посчитанный с помощью метода золотого сечения
def golden_ratio_step(step_size, x_last, function, gradient, eps):
    phi = (1 + 5 ** 0.5) / 2
    a = 0
    b = 1
    ml = b - (b - a) / phi
    mr = a + (b - a) / phi
    xl = [x_i - ml * grad_i for x_i, grad_i in zip(x_last, gradient)]
    xr = [x_i - mr * grad_i for x_i, grad_i in zip(x_last, gradient)]
    lvalue = function(xl)
    rvalue = function(xr)
    cnt_func = 2

    while b - a > eps:
        if lvalue < rvalue:
            b = mr
            mr = ml
            ml = b - (b - a) / phi
            rvalue = lvalue
            xr = xl
            xl = [x - ml * grad_i for x, grad_i in zip(x_last, gradient)]
            lvalue = function(xl)
            cnt_func += 1
        else:
            a = ml
            ml = mr
            mr = a + (b - a) / phi
            lvalue = rvalue
            xl = xr
            xr = [x - mr * grad_i for x, grad_i in zip(x_last, gradient)]
            rvalue = function(xr)
            cnt_func += 1

    return np.array([a, cnt_func])


# Функция возвращает шаг, посчитанный с помощью метода деления пополам
def bisection(step_size, x_last, function, gradient, eps):
    a = 0
    b = 1
    cnt_func = 0

    while b - a > 3 * eps:
        ml = (a + b) / 2 - eps
        mr = (a + b) / 2 + eps
        xl = [x_i - ml * grad_i for x_i, grad_i in zip(x_last, gradient)]
        xr = [x_i - mr * grad_i for x_i, grad_i in zip(x_last, gradient)]
        cnt_func+=2
        if function(xl) <= function(xr):
            b = mr
        else:
            a = ml

    return np.array([a, cnt_func])


# Градиентный спуск
def gradient_descent(function, start_point, step_function, step_size, eps):
    x_res = start_point
    x_last = start_point
    cnt_grad = 0
    cnt_func = 0
    cnt_iter = 0

    while True:
        cnt_iter += 1

        # Вычисление градиента
        cnt_grad += 1
        gradient = sp.optimize.approx_fprime(x_last, function, eps)

        # Обновление точки
        eval_step = step_function(step_size, x_last, function, gradient, eps)
        step = eval_step[0]
        cnt_func += eval_step[1]
        x_new = [x - step * grad_i for x, grad_i in zip(x_last, gradient)]

        # Условие выхода
        cnt_func += 2
        if abs(function(x_new) - function(x_last)) < eps:
            break

        x_last = x_res
        x_res = x_new

    return np.array([x_res[0], x_res[1], function(x_res), cnt_iter, cnt_grad, cnt_func])


# Градиентный спуск с константным шагом
def gradient_descent_with_constant_step(function, start_point, step_size, eps):
    return gradient_descent(function, start_point, constant_step, step_size, eps)

# Градиентный спуск с шагом, посчитанным с помощью метода золотого сечения
def gradient_descent_with_golden_ratio_step(function, start_point, eps):
    return gradient_descent(function, start_point, golden_ratio_step, None, eps)

# Градиентный спуск с шагом, посчитанным с помощью метода деления пополам
def gradient_descent_with_bisection(function, start_point, eps):
    return gradient_descent(function, start_point, bisection, None, eps)


def message(result):
    print("----- Argument values = [", result[0], result[1], "]")
    print("----- Function result = ", result[2])
    print("----- Iteration count = ", result[3])
    print("----- Gradient evaluation count = ", result[4])
    print("----- Function evaluation count = ", result[5])
    print()

def test(start_point, function, const_step, eps):
    print("Start point: " + str(start_point))
    print("Constant step: ", const_step)
    print("Required accuracy: ", eps)

    Nelder_Mead_res = sp.optimize.minimize(function, start_point, method="Nelder-Mead")
    print("Nelder-Mead from scipy.optimize")
    message(np.array([Nelder_Mead_res.x[0], Nelder_Mead_res.x[1], function(Nelder_Mead_res.x), Nelder_Mead_res.nit, 0, Nelder_Mead_res.nfev]))

    constant_gradient_res = gradient_descent_with_constant_step(function, start_point, const_step, eps)
    print("Gradient descent with constant step")
    message(constant_gradient_res)

    golden_ratio_res = gradient_descent_with_golden_ratio_step(function, start_point, eps)
    print("Gradient descent with golden ratio step")
    message(golden_ratio_res)

    bisection_res = gradient_descent_with_bisection(function, start_point, eps)
    print("Gradient descent with bisection step")
    message(bisection_res)


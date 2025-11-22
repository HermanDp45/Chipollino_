#include "Objects/FiniteAutomaton.h"
#include "Objects/Language.h"
#include "Objects/iLogTemplate.h"
#include <Objects/BackRefRegex.h>
#include <Objects/PushdownAutomaton.h>
#include <Objects/Symbol.h>
#include <sstream>
#include <string>

using std::optional;
using std::pair;
using std::set;
using std::stack;
using std::string;
using std::stringstream;
using std::tuple;
using std::unordered_map;
using std::unordered_set;
using std::vector;

template <typename Range, typename Value = typename Range::value_type>

/**
 * @brief Объединяет элементы диапазона в строку с разделителем.
 * @param elements диапазон элементов для объединения.
 * @param delimiter разделитель между элементами.
 * @return строка с объединенными элементами.
*/
std::string Join(Range const& elements, const char* const delimiter) {
	std::ostringstream os;
	auto b = begin(elements), e = end(elements);

	if (b != e) {
		std::copy(b, prev(e), std::ostream_iterator<Value>(os, delimiter));
		b = prev(e);
	}
	if (b != e) {
		os << *b;
	}

	return os.str();
}

// redefinition of 'hash_combine' in Tools.h
// template <typename T> void hash_combine(std::size_t& seed, const T& v) {
// 	seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
// }

/**
 * @brief Конструктор перехода PDA.
 * @param to индекс целевого состояния.
 * @param input входной символ.
 * @param pop символ, извлекаемый из стека.
 * @param push вектор символов для помещения в стек.
 */
PDATransition::PDATransition(const int to, const Symbol& input, const Symbol& pop,
							 const std::vector<Symbol>& push)
	: to(to), input_symbol(input), push(push), pop(pop) {}

bool PDATransition::operator==(const PDATransition& other) const {
	return to == other.to && input_symbol == other.input_symbol && push == other.push &&
		   pop == other.pop;
}

/**
 * @brief Функтор для хэширования PDATransition.
 * @param t переход для хэширования.
 * @return хэш-значение.
 */
std::size_t PDATransition::Hasher::operator()(const PDATransition& t) const {
	std::size_t hash = 0;
	hash_combine(hash, t.to);
	hash_combine(hash, string(t.input_symbol));
	hash_combine(hash, string(t.pop));
	for (const auto& push_symb : t.push) {
		hash_combine(hash, string(push_symb));
	}
	return hash;
}

/**
 * @brief Конструктор состояния PDA.
 * @param index индекс состояния.
 * @param is_terminal является ли состояние терминальным.
 */
PDAState::PDAState(int index, bool is_terminal) : State(index, {}, is_terminal) {}

/**
 * @brief Конструктор состояния PDA с идентификатором.
 * @param index индекс состояния.
 * @param identifier идентификатор состояния.
 * @param is_terminal является ли состояние терминальным.
 */
PDAState::PDAState(int index, string identifier, bool is_terminal)
	: State(index, std::move(identifier), is_terminal) {}

/**
 * @brief Конструктор состояния PDA с переходами.
 * @param index индекс состояния.
 * @param identifier идентификатор состояния.
 * @param is_terminal является ли состояние терминальным.
 * @param transitions переходы состояния.
 */
PDAState::PDAState(int index, std::string identifier, bool is_terminal, Transitions transitions)
	: State(index, std::move(identifier), is_terminal), transitions(std::move(transitions)) {}

/**
 * @brief Преобразует состояние в текстовый формат.
 * @return пустая строка (пока не готово или еще не осознал).
 */
std::string PDAState::to_txt() const {
	return {};
}

/**
 * @brief Добавляет переход в состояние (без полной информации).
 * @param to переход.
 * @param input_symbol входной символ.
 */
void PDAState::set_transition(const PDATransition& to, const Symbol& input_symbol) {
	transitions[input_symbol].insert(to);
}

/**
 * @brief Конструктор PDA по умолчанию.
 */
PushdownAutomaton::PushdownAutomaton() : AbstractMachine() {}

/**
 * @brief Конструктор PDA с состояниями и алфавитом.
 * @param initial_state начальное состояние.
 * @param states вектор состояний.
 * @param alphabet алфавит.
 */
PushdownAutomaton::PushdownAutomaton(int initial_state, std::vector<PDAState> states,
									 Alphabet alphabet)
	: AbstractMachine(initial_state, std::move(alphabet)), states(std::move(states)) {
	for (int i = 0; i < this->states.size(); i++) {
		if (this->states[i].index != i)
			throw std::logic_error(
				"State.index must correspond to its ordinal number in the states vector");
	}
}

/**
 * @brief Конструктор PDA с состояниями и языком.
 * @param initial_state начальное состояние.
 * @param states вектор состояний.
 * @param language язык.
 */
PushdownAutomaton::PushdownAutomaton(int initial_state, vector<PDAState> states,
									 std::shared_ptr<Language> language)
	: AbstractMachine(initial_state, std::move(language)), states(std::move(states)) {
	// проходимся по состояниям и проверяем индексы
	for (int i = 0; i < this->states.size(); i++) {
		if (this->states[i].index != i)
			throw std::logic_error(
				"State.index must correspond to its ordinal number in the states vector");
	}
}

/**
 * @brief Преобразует PDA в формат DOT для GraphViz.
 * @return строка в формате DOT.
 */
std::string PushdownAutomaton::to_txt() const {
	stringstream ss;
	ss << "digraph {\n\trankdir = LR\n\tdummy [label = \"\", shape = none]\n\t";
	for (const auto& state : states) {
		ss << state.index << " [label = \"" << state.identifier << "\", shape = ";
		ss << (state.is_terminal ? "doublecircle]\n\t" : "circle]\n\t");
	}
	if (states.size() > initial_state)
		ss << "dummy -> " << states[initial_state].index << "\n";

	for (const auto& state : states) {
		for (const auto& elem : state.transitions) {
			for (const auto& transition : elem.second) {
				ss << "\t" << state.index << " -> " << transition.to << " [label = \""
				   << string(elem.first) << ", " << transition.pop << "/"
				   << (transition.push.empty() ? "eps" : Join(transition.push, ",")) << "\"]\n";
			}
		}
	}

	ss << "}\n";
	return ss.str();
}

/**
 * @brief Возвращает вектор состояний PDA.
 * @return Вектор состояний.
 */
std::vector<PDAState> PushdownAutomaton::get_states() const {
	return states;
}

/**
 * @brief Возвращает количество состояний PDA.
 * @param log логгер.
 * @return Количество состояний.
 */
size_t PushdownAutomaton::size(iLogTemplate* log) const {
	return states.size();
}

/**
 * @brief Удаляет недостижимые состояния из PDA.
 * @param log логгер.
 * @return Новый PDA без недостижимых состояний.
 */
PushdownAutomaton PushdownAutomaton::_remove_unreachable_states(iLogTemplate* log) {
	if (states.size() == 1) {
		return *this;
	}

	std::unordered_set<int> reachable_states = {initial_state};
	std::queue<int> states_to_visit;
	states_to_visit.push(initial_state);

	// Perform BFS to find reachable states
	while (!states_to_visit.empty()) {
		int current_state_index = states_to_visit.front();
		states_to_visit.pop();
		const PDAState& current_state = states[current_state_index];

		// Visit transitions from the current state
		for (const auto& [symbol, symbol_transitions] : current_state.transitions) {
			for (const auto& trans : symbol_transitions) {
				int index = trans.to;
				// If the next state is not already marked as reachable
				if (!reachable_states.count(index)) {
					reachable_states.insert(index);
					states_to_visit.push(index);
				}
			}
		}
	}

	// Remove unreachable states from the states vector
	std::vector<PDAState> new_states;
	int i = 0, new_initial_index;
	std::unordered_map<int, int> recover_index;
	for (auto state : states) {
		if (!reachable_states.count(state.index)) {
			continue;
		}

		if (state.index == initial_state) {
			new_initial_index = i;
		}
		recover_index[state.index] = i;
		state.index = i;
		new_states.emplace_back(state);
		i++;
	}

	// Remap transitions
	for (auto& state : new_states) {
		PDAState::Transitions new_transitions;
		for (const auto& [symbol, symbol_transitions] : state.transitions) {
			for (const auto& trans : symbol_transitions) {
				new_transitions[symbol].insert(PDATransition(
					recover_index[trans.to], trans.input_symbol, trans.pop, trans.push));
			}
		}
		state.transitions = new_transitions;
	}

	return {new_initial_index, new_states, get_language()};
}

/**
 * @brief Вычисляет пересечение PDA с регулярным выражением.
 * @param re регулярное выражение.
 * @param log логгер.
 * @return новый PDA, представляющий пересечение.
 */
PushdownAutomaton PushdownAutomaton::regular_intersect(const Regex& re, iLogTemplate* log) {
	// преобразуем регулярное выражение в недетерминированный конечный автомат (NFA)
	// детерминизируем - DFA
	// минимизируем число состояний (оптимизируем)
	auto dfa = re.to_thompson(log).determinize(log).minimize(log);

	// Flatten PDA transitions
	// разворачиваем переходы PDA в список вида [(откуда, переходы)]
	using PDA_from = int;
	std::vector<std::pair<PDA_from, PDATransition>> pda_transitions_by_from;
	for (const auto& state : states) {
		for (const auto& [symbol, symbol_transitions] : state.transitions) {
			for (const auto& trans : symbol_transitions) {
				pda_transitions_by_from.emplace_back(state.index, trans);
			}
		}
	}

	// Initialize states.
	// recover_index maps (pda_index, dfa_index) to result_index.
	std::vector<PDAState> result_states;
	std::unordered_map<std::pair<int, int>, int, PairHasher<int,int>> recover_index; // fix template form
	int i = 0;
	// через комбинации формируем новые состояния вида (q_i,q_j)
	// терминальность в случае одновременной терминальности q_i,q_j
	for (const auto& pda_state : states) {
		for (const auto& dfa_state : dfa.get_states()) {
			std::ostringstream oss;
			oss << "(" << pda_state.identifier << ";" << dfa_state.identifier << ")";
			result_states.emplace_back(
				i, oss.str(), pda_state.is_terminal && dfa_state.is_terminal);
			recover_index[{pda_state.index, dfa_state.index}] = i;
			i++;
		}
	}

	// Calculate transitions
	for (const auto& [pda_from, pda_trans] : pda_transitions_by_from) {
		for (const auto& dfa_state : dfa.get_states()) {
			// Process epsilon transitions separately
			// если в PDA был eps-переход, то добавляем его с неизменным состоянием для DFA 
			if (pda_trans.input_symbol.is_epsilon()) {
				auto from_index = recover_index[{pda_from, dfa_state.index}]; 
				auto to_index = recover_index[{pda_trans.to, dfa_state.index}];
				result_states[from_index].set_transition(
					PDATransition(to_index, pda_trans.input_symbol, pda_trans.pop, pda_trans.push),
					pda_trans.input_symbol);
				continue;
			}

			// Process regular transitions separately
			// берем символ, ищем переходы через него в dfa, формируем новый объединенный переход
			auto matching_transitions = dfa_state.transitions.find(pda_trans.input_symbol);
			if (matching_transitions == dfa_state.transitions.end()) {
				continue;
			}

			for (const auto& dfa_to : matching_transitions->second) {
				auto from_index = recover_index[{pda_from, dfa_state.index}];
				auto to_index = recover_index[{pda_trans.to, dfa_to}];
				result_states[from_index].set_transition(
					PDATransition(to_index, pda_trans.input_symbol, pda_trans.pop, pda_trans.push),
					pda_trans.input_symbol);
			}
		}
	}

	auto result_initial_state = recover_index[{initial_state, dfa.get_initial()}];

	// Calculate resulting alphabet
	// просто пересечение алфавитов
	auto pda_alphabet = get_language()->get_alphabet();
	auto dfa_alphabet = dfa.get_language()->get_alphabet();
	Alphabet result_alphabet;
	std::set_intersection(pda_alphabet.begin(),
						  pda_alphabet.end(),
						  dfa_alphabet.begin(),
						  dfa_alphabet.end(),
						  std::inserter(result_alphabet, result_alphabet.begin()));

	auto result = PushdownAutomaton(result_initial_state, result_states, result_alphabet);
	result = result._remove_unreachable_states(log);

	if (log) {
		log->set_parameter("pda", *this);
		log->set_parameter("regex", dfa);
		log->set_parameter("result", result);
	}

	return result;
}

/**
 * @brief Преобразует переходы состояния в плоский вектор.
 * @param state состояние.
 * @return вектор переходов.
 */
vector<PDATransition> _flatten_transitions(PDAState state) {
	vector<PDATransition> result;

	for (const auto& [symbol, symbol_transitions] : state.transitions) {
		for (const auto& trans : symbol_transitions) {
			result.emplace_back(trans);
		}
	}

	return result;
}

/**
 * @brief Рекурсивная функция для проверки равенства PDA.
 * @param pda1 первый PDA.
 * @param pda2 второй PDA.
 * @param chst состояние проверки.
 * @return Пара: результат и обновленное состояние.
 */
pair<bool, EqualityCheckerState> PushdownAutomaton::_equality_dfs(
	const PushdownAutomaton& pda1, const PushdownAutomaton& pda2, EqualityCheckerState chst) const {

	if (!chst.can_map_states(chst.index1, chst.index2) ||
		pda1.states[chst.index1].is_terminal != pda2.states[chst.index2].is_terminal) {
		return {false, {}};
	} else if (chst.states_mapping.count(chst.index1) && chst.states_mapping[chst.index1] == chst.index2) {
		return {true, chst};
	}
	chst.map_states(chst.index1, chst.index2);

	const auto state1 = pda1.states[chst.index1];
	const auto state2 = pda2.states[chst.index2];

	if (state1.transitions.size() != state2.transitions.size()) {
		// Два состояния должны иметь переходы по равному количеству символов.
		return {false, {}};
	}
	// мне кажется, что еще надо смотреть на сами символы, а не только их количество

	auto transitions1 = _flatten_transitions(state1);
	auto transitions2 = _flatten_transitions(state2);
	if (transitions1.size() != transitions2.size()) {
		return {false, {}};
	}

	struct st {
		int rem1_index;
		vector<PDATransition> rem1, rem2;
		EqualityCheckerState chst;
	};

	std::queue<st> q;
	q.push({0, transitions1, transitions2, chst});
	while (!q.empty()) {
		auto data = q.front();
		q.pop();

		if (data.rem1_index >= data.rem1.size()) {
			return {true, data.chst};
		}

		auto trans1 = data.rem1[data.rem1_index];
		for (int i = 0; i<data.rem2.size(); i++) {
			auto chst_copy = data.chst;
			auto trans2 = data.rem2[i];

			if(trans1.input_symbol != trans2.input_symbol) {
				continue;
			}

			// Проверяем возможность сопоставить pop символы
			if (!chst_copy.can_map_stack(trans1.pop, trans2.pop))
				continue;
			chst_copy.map_stack(trans1.pop, trans2.pop);

			// Проверяем возможность сопоставить push символы
			if (trans1.push.size() != trans2.push.size())
				continue;

			bool is_push_mapped = true;
			for (int i = 0; i < trans1.push.size(); i++) {
				auto ss1 = trans1.push[i], ss2 = trans2.push[i];
				if (!chst_copy.can_map_stack(ss1, ss2)) {
					is_push_mapped = false;
					break;
				}
				chst_copy.map_stack(ss1, ss2);
			}
			if (!is_push_mapped)
				continue;

			auto [success, chst_new] = _equality_dfs(pda1, pda2, {trans1.to, trans2.to, chst_copy.stack_mapping, chst_copy.states_mapping});
			if (!success) {
				continue;
			}

			// Удалось сопоставить переход, пробуем раскрывать дальше
			data.rem2.erase(data.rem2.begin()+i);
			q.push({data.rem1_index+1, data.rem1, data.rem2, chst_new});
		}
	}

	return {false, {}};
}

/**
 * @brief Проверяет равенство двух PDA.
 * @param pda1 первый PDA.
 * @param pda2 второй PDA.
 * @return Пара: результат и отображение стековых символов.
 */
pair<bool, unordered_map<Symbol, Symbol, Symbol::Hasher>> PushdownAutomaton::_equality_checker(PushdownAutomaton pda1, PushdownAutomaton pda2) {
	// Автоматы должны как минимум иметь равное количество состояний,
	// равное количество символов стэка и равные алфавиты.
	if (pda1.size() != pda2.size() ||
		pda1.get_language()->get_alphabet() != pda2.get_language()->get_alphabet() ||
		pda1._get_stack_symbols().size() != pda2._get_stack_symbols().size()) {
		return {false, {}};
	}

	const EqualityCheckerState chst(pda1.initial_state,
									pda2.initial_state,
									{{Symbol::StackTop, Symbol::StackTop}, {Symbol::Epsilon, Symbol::Epsilon}},
									{});
	const auto [success, chst_new] = pda1._equality_dfs(pda1, pda2, chst);
	if (!success) {
		return {false, {}};
	}

	return {true, chst_new.stack_mapping};
}

/**
 * @brief Проверяет равенство двух PDA.
 * @param pda1 первый PDA.
 * @param pda2 второй PDA.
 * @param log логгер.
 * @return true, если PDA равны.
 */
bool PushdownAutomaton::equal(PushdownAutomaton pda1, PushdownAutomaton pda2, iLogTemplate* log) {
	auto [result, stack_mapping] = _equality_checker(pda1, pda2);

	if (log) {
		log->set_parameter("pda1", pda1);
		log->set_parameter("pda2", pda2);
		log->set_parameter("result", result);
		if (!result) {
			log->set_parameter("stack", "{}");
		} else {
			std::ostringstream oss;
			for (auto [first, second]: stack_mapping) {
				oss << first << " -> " << second << "; ";
			}
			log->set_parameter("stack", oss.str());
		}
	}

	return result;
}

/**
 * @brief Проверяет, является ли PDA детерминированным.
 * @param log логгер.
 * @return true, если PDA детерминирован.
 */
bool PushdownAutomaton::is_deterministic(iLogTemplate* log) const {
	bool result = true;
	std::unordered_set<int> nondeterministic_states;
	for (const auto& state : states) {
		// Отображение символов стэка в символы алфавита
		std::unordered_map<Symbol, std::unordered_set<Symbol, Symbol::Hasher>, Symbol::Hasher>
			stack_sym_to_sym;
		for (const auto& [symb, symbol_transitions] : state.transitions) {
			for (const auto& tr : symbol_transitions) {
				// Если помимо эпсилон перехода будет иной переход с тем же символом стэка,
				// то автомату не понятно, считывать следующий или нет - недетерминированность.
				if (symb.is_epsilon() && !stack_sym_to_sym[tr.pop].empty()) {
					// Переход по эпсилону с pop некоторого символа стэка.
					// С этим символом стэка не должно быть иных переходов.
					result = false;
					nondeterministic_states.emplace(state.index);
					break;
				}

				if (stack_sym_to_sym[tr.pop].count(symb) ||
					stack_sym_to_sym[tr.pop].count(Symbol::Epsilon)) {
					// Перехода по одной и той же паре (symb, stack_symb) не должно быть.
					// Так же не может быть перехода по символу стэка, для которого ранее
					// зафиксировали наличие эпсилон-перехода.
					result = false;
					nondeterministic_states.emplace(state.index);
					break;
				}

				stack_sym_to_sym[tr.pop].emplace(symb);
			}
		}
	}

	if (log) {
		MetaInfo meta;
		for (const auto& state : states) {
			if (nondeterministic_states.count(state.index)) {
				meta.upd(NodeMeta{state.index, MetaInfo::trap_color});
			}
		}
		log->set_parameter("pda", *this, meta);
		log->set_parameter("result", result ? "True" : "False");
	}

	return result;
}

/**
 * @brief Возвращает множество стековых символов PDA.
 * @return Множество символов.
 */
std::unordered_set<Symbol, Symbol::Hasher> PushdownAutomaton::_get_stack_symbols() const {
	std::unordered_set<Symbol, Symbol::Hasher> result;
	for (const auto& state : states) {
		for (const auto& [_, symbol_transitions] : state.transitions) {
			for (const auto& trans : symbol_transitions) {
				if (!trans.pop.is_epsilon()) {
					result.emplace(trans.pop);
				}
			}
		}
	}
	return result;
}

/**
 * @brief DFS для поиска достижимых состояний по epsilon-переходам.
 * @param index индекс состояния.
 * @param reachable Множество достижимых состояний.
 */
void PushdownAutomaton::_dfs(int index, unordered_set<int>& reachable) const {
	if (reachable.count(index)) {
		return;
	}
	reachable.insert(index);

	const auto& by_eps = states[index].transitions.find(Symbol::Epsilon);
	if (by_eps == states[index].transitions.end()) {
		return;
	}

	for (auto& trans_to : by_eps->second) {
		_dfs(trans_to.to, reachable);
	}
}

/**
 * @brief Вычисляет замыкание eps-замыкание для состояния.
 * @param index индекс состояния.
 * @return Отображение переходов на достижимые состояния.
 */
unordered_map<PDATransition, unordered_set<int>, PDATransition::Hasher> PushdownAutomaton::closure(
	const int index) const {
	unordered_map<PDATransition, unordered_set<int>, PDATransition::Hasher> result;

	auto state = states[index];
	if (state.transitions.find(Symbol::Epsilon) == state.transitions.end()) {
		return result;
	}

	for (const auto& eps_trans : state.transitions.at(Symbol::Epsilon)) {
		unordered_set<int> reachable;
		_dfs(eps_trans.to, reachable);
		result[eps_trans] = reachable;
	}

	return result;
}

/**
 * @brief Находит проблемные eps-переходы для дополнения.
 * @return Отображение состояний на проблемные переходы.
 */
std::unordered_map<int, std::unordered_set<PDATransition, PDATransition::Hasher>>
PushdownAutomaton::_find_problematic_epsilon_transitions() const {
	std::unordered_map<int, std::unordered_set<PDATransition, PDATransition::Hasher>> result;

	std::unordered_set<int> states_with_problematic_trans;
	for (const auto& state : states) {
		// Ищем нефинальные состояния, из которых есть помимо eps-переходов есть иные переходы.
		if (state.is_terminal ||
			state.transitions.find(Symbol::Epsilon) == state.transitions.end()) {
			continue;
		}

		// Отмечаем все eps-переходы в финальные состояния как проблемные.
		for (const auto& trans : state.transitions.at(Symbol::Epsilon)) {
			if (states[trans.to].is_terminal) {
				result[state.index].emplace(trans);
				states_with_problematic_trans.emplace(state.index);
			}
		}
	}

	for (const auto& state : states) {
		if (state.is_terminal ||
			state.transitions.find(Symbol::Epsilon) == state.transitions.end()) {
			continue;
		}

		auto reachable = closure(state.index);
		// Ищем нефинальные состояния, из которых есть помимо eps-переходов есть иные переходы.
		for (const auto& [eps_trans, indices] : reachable) {
			for (const auto& index : indices) {
				if (states[index].is_terminal) {
					result[state.index].emplace(eps_trans);
				}
			}
		}
	}

	return result;
}

/**
 * @brief Находит все переходы, ведущие в заданное состояние.
 * @param index индекс состояния.
 * @return Вектор пар (откуда, переход).
 */
std::vector<std::pair<int, PDATransition>> PushdownAutomaton::_find_transitions_to(
	int index) const {
	std::vector<std::pair<int, PDATransition>> result;

	for (const auto& state : states) {
		for (const auto& [symbol, symbol_transitions] : state.transitions) {
			for (const auto& trans : symbol_transitions) {
				if (trans.to == index) {
					result.emplace_back(state.index, trans);
				}
			}
		}
	}

	return result;
}

/**
 * @brief Добавляет ловушку-состояние в PDA.
 * @return Новый PDA с ловушкой.
 */
PushdownAutomaton PushdownAutomaton::_add_trap_state() {
	// копируем автомат
	PushdownAutomaton result(initial_state, states, language);
	// получаем все символы, которые в переходах снимались со стека
	auto stack_symbols = result._get_stack_symbols();
	// инициализируем индекс ловушки (либо берем уже имеющейся)
	int i_trap = static_cast<int>(result.size());
	bool already_has_trap = false;
	
	// Ловушка удовлетворяет следующей семантике: 
	// 1) нефинальное состояние
	// 2) все переходы ведут в само себя
	// 3) переходы не меняют стек
	for (const auto& state : result.states) {
		// состояние-ловушка должно быть нефинальным
		if (state.is_terminal) {
			continue;
		}
		
		// проверяем, является ли состояние ловушкой
		bool is_trap = true;
		for (const auto& [symbol, symbol_transitions] : state.transitions) {
			for (const auto& trans : symbol_transitions) {
				// все переходы должны вести в само состояние
				if (trans.to != state.index) {
					is_trap = false;
					break;
				}
				// переход не должен менять стек: снимаем X, кладем обратно [X]
				if (trans.push.size() != 1 || trans.push[0] != trans.pop) {
					is_trap = false;
					break;
				}
			}
			if (!is_trap) {
				break;
			}
		}
		
		if (is_trap && !state.transitions.empty()) {
			already_has_trap = true;
			i_trap = state.index;
			break;
		}
	}

	bool need_create_trap = false;
	// проходимся по каждому состоянию
	for (auto& state : result.states) {
		std::set<std::pair<Symbol, Symbol>> stack_sym_in_sym_transitions;
		// проходим по всем символам и переходам по ним в состоянии
		for (const auto& [symbol, symbol_transitions] : state.transitions) {
			// проходим каждому переходу по символу
			for (const auto& trans : symbol_transitions) {
				// добавляем в общее множество элемент ("Что снимаем со стека", "по какому символу переходи")
				stack_sym_in_sym_transitions.emplace(trans.pop, trans.input_symbol);
			}
		}
		
		// Получаем все символы алфавита языка
		std::set<Symbol> symbols = result.get_language()->get_alphabet();

		for (const auto& symb : symbols) {
			for (const auto& stack_symb : stack_symbols) {
				if (stack_sym_in_sym_transitions.count({stack_symb, symb}) ||
					stack_sym_in_sym_transitions.count({stack_symb, Symbol::Epsilon})) {
					continue;
				}
				need_create_trap = true;
				state.set_transition({i_trap, symb, stack_symb, std::vector<Symbol>{stack_symb}},
									 symb);
			}
		}
	}

	// Если не добавили не одного состояния или состояние ловушки уже было
	// то заканчиваем 
	if (!need_create_trap || already_has_trap) {
		return result;
	}

	// иначе добавляем состояние ловушки
	result.states.emplace_back(i_trap, "trap", false);
	// добавляем замыкание ловушки на самом себе
	for (const auto& symb : result.get_language()->get_alphabet()) {
		if (symb.is_epsilon())
			continue;

		for (const auto& stack_symb : stack_symbols) {
			result.states[i_trap].set_transition(
				{i_trap, symb, stack_symb, std::vector<Symbol>{stack_symb}}, symb);
		}
	}

	return result;
}

/**
 * @brief Вычисляет дополнение PDA.
 * @param log логгер.
 * @return Дополнение PDA.
 */
PushdownAutomaton PushdownAutomaton::complement(iLogTemplate* log) const {
	// PDA here is deterministic.
	if (!is_deterministic()) {
		throw std::logic_error("Complement is available only for deterministic PDA");
	}

	PushdownAutomaton result(initial_state, states, language->get_alphabet());
	result = result._add_trap_state();

	std::set<int> no_toggle_states;
	auto problematic_transitions = result._find_problematic_epsilon_transitions();
	for (const auto& [from_index, bad_transitions] : problematic_transitions) {
		for (const auto& bad_trans : bad_transitions) {
			auto bad_symbol = bad_trans.pop;
			auto final_state_index = bad_trans.to;
			auto problems_trap_index = static_cast<int>(result.size());
			no_toggle_states.emplace(
				problems_trap_index); // Состояние-ловушка проблемных переходов не
			// меняет финальность при обращении

			for (const auto& [from_from_index, trans] : result._find_transitions_to(from_index)) {
				if (!trans.push.empty() && trans.push.back() == bad_symbol) {
					// Если после перехода на вершине стэка точно окажется "проблемный символ", то
					// перенаправляем переход сразу в финальное состояние.
					result.states[from_from_index].set_transition(
						{final_state_index, trans.input_symbol, trans.pop, trans.push},
						trans.input_symbol);
					result.states[from_from_index].transitions[trans.input_symbol].erase(trans);
					continue;
				}

				// Иначе перенаправляем переход в ловушку проблемных переходов
				result.states[from_from_index].set_transition(
					{problems_trap_index, trans.input_symbol, trans.pop, trans.push},
					trans.input_symbol);
				result.states[from_from_index].transitions[trans.input_symbol].erase(trans);
			}

			// Добавляем состояние-ловушку проблемных переходов.
			result.states.emplace_back(problems_trap_index, "eps-trap", false);
			result.states[problems_trap_index].set_transition(
				{final_state_index, Symbol::Epsilon, bad_symbol, std::vector<Symbol>({bad_symbol})},
				Symbol::Epsilon);
			for (const auto& stack_symb : result._get_stack_symbols()) {
				if (stack_symb != bad_symbol) {
					result.states[problems_trap_index].set_transition(
						{from_index, Symbol::Epsilon, stack_symb, std::vector<Symbol>({stack_symb})},
						Symbol::Epsilon);
				}
			}
		}
	}

	result = result._add_trap_state();
	for (auto& state : result.states) {
		if (!no_toggle_states.count(state.index)) {
			state.is_terminal = !state.is_terminal;
		}
	}

	if (log) {
		log->set_parameter("oldpda", *this);
		log->set_parameter("result", result);
	}
	return result;
}

/**
 * @brief Получает регулярные переходы для состояния разбора.
 * @param s входная строка.
 * @param parsing_state текущее состояние разбора.
 * @return Вектор регулярных переходов.
 */
std::vector<PDATransition> get_regular_transitions(const string& s,
												   const ParsingState& parsing_state) {
	std::vector<PDATransition> regular_transitions;
	const auto& transitions = parsing_state.state->transitions;
	
	// получаем текщий символ
	const Symbol symb(parsing_state.pos < s.size() ? s[parsing_state.pos] : char());
	// если нет по этому символу перехода, то возвращаем пустой
	if (transitions.find(symb) == transitions.end()) {
		return regular_transitions;
	}

	// получаем все переходы по этому символу и берем только те,
	// где символ стека перехода совпадает с вершиной стека
	auto symbol_transitions = transitions.at(symb);
	for (const auto& trans : symbol_transitions) {
		if (!parsing_state.stack.empty() && parsing_state.stack.top() == trans.pop) {
			regular_transitions.emplace_back(trans);
		}
	}

	return regular_transitions;
}

/**
 * @brief Получает epsilon-переходы для состояния разбора.
 * @param parsing_state текущее состояние разбора.
 * @return Вектор epsilon-переходов.
 */
std::vector<PDATransition> get_epsilon_transitions(const ParsingState& parsing_state) {
	std::vector<PDATransition> epsilon_transitions;
	const auto& transitions = parsing_state.state->transitions;

	if (transitions.find(Symbol::Epsilon) == transitions.end()) {
		return epsilon_transitions;
	}

	for (const auto& trans : transitions.at(Symbol::Epsilon)) {
		// переход по epsilon -> не потребляем символ
		if (!parsing_state.stack.empty() && parsing_state.stack.top() == trans.pop) {
			epsilon_transitions.emplace_back(trans);
		}
	}

	return epsilon_transitions;
}

/**
 * @brief Выполняет действия со стеком для перехода.
 * @param stack текущий стек.
 * @param tr переход.
 * @return Новый стек после действий.
 */
std::stack<Symbol> perform_stack_actions(std::stack<Symbol> stack, const PDATransition& tr) {
	stack.pop();

	for (const auto& push_sym : tr.push) {
		if (!push_sym.is_epsilon()) {
			stack.push(push_sym);
		}
	}
	return stack;
}

/**
 * @brief Вычисляет хеш содержимого стека.
 * @param stack стек для хеширования (копия, не изменяем оригинал).
 * @return хеш-значение.
 */
static size_t hash_stack(const std::stack<Symbol>& stack) {
	size_t hash = 0;
	// Копируем стек для обхода
	std::stack<Symbol> temp_stack = stack;
	std::vector<std::string> symbols_as_strings;
	
	while (!temp_stack.empty()) {
		symbols_as_strings.push_back(string(temp_stack.top()));
		temp_stack.pop();
	}
	
	// Хешируем в обратном порядке для консистентности
	for (const auto& sym_str : symbols_as_strings) {
		hash_combine(hash, sym_str);
	}
	return hash;
}

/**
 * @brief Разбирает строку с помощью PDA.
 * @param s входная строка.
 * @return Пара: счетчик шагов и результат разбора.
 */
std::pair<int, bool> PushdownAutomaton::parse(const std::string& s) const {
	// сounter - счетчик итераций цикла
	// parsed_len - сколько элементов строки уже обработали
	int counter = 0, parsed_len = 0;
	// берем стартовое состояние
	const PDAState* state = &states[initial_state];
	
	// Мемоизация полных конфигураций (state_index, pos, stack_hash)
	// чтобы избежать повторной обработки одинаковых состояний
	set<tuple<int, int, size_t>> visited_configs;
	
	// стэк PDA
	std::stack<Symbol> pda_stack;
	pda_stack.emplace(Symbol::StackTop);
	// Стек того, что необходимо еще обработать
	// так как вариантов обхода много - обходим все таким образом
	std::stack<ParsingState> parsing_stack;
	parsing_stack.emplace(parsed_len, state, pda_stack);

	while (!parsing_stack.empty()) {
		// Если дошли до конца и при этом в финальном состоянии
		// заканчиваем проход
		if (state->is_terminal && parsed_len == s.size()) {
			break;
		}
		
		// достаем состояние разбора со стека
		auto parsing_state = parsing_stack.top();
		parsing_stack.pop();

		// получаем для него состояние
		state = parsing_state.state;
		// уже обработанную длину
		parsed_len = parsing_state.pos;
		// текущий стек
		pda_stack = parsing_state.stack;
		
		// Проверяем, не посещали ли мы уже эту конфигурацию
		size_t stack_hash = hash_stack(pda_stack);
		auto config = std::make_tuple(state->index, parsed_len, stack_hash);
		if (visited_configs.count(config)) {
			continue; // Уже обработали эту конфигурацию
		}
		visited_configs.insert(config);
		
		// добавляем общую итерацию
		counter++;
		
		// получаем все не eps переходы для текущего символа и состояния
		auto transitions = get_regular_transitions(s, parsing_state);
		// для каждого такого перехода добавляем в стек следующие варианты разборы
		if (parsed_len + 1 <= s.size()) {
			for (const auto& trans : transitions) {
				parsing_stack.emplace(parsed_len + 1,
									  &states[trans.to],
									  perform_stack_actions(parsing_state.stack, trans));
			}
		}

		// добавление эпсилон-переходов
		auto eps_transitions = get_epsilon_transitions(parsing_state);
		for (const auto& trans : eps_transitions) {
			parsing_stack.emplace(parsed_len, &states[trans.to], perform_stack_actions(parsing_state.stack, trans));
		}
	}

	// если дошли ровно до конца и в финальном состоянии, то успех
	if (s.size() == parsed_len && state->is_terminal) {
		return {counter, true};
	}

	// или не успех
	return {counter, false};
}

/**
 * @brief Возвращает обратные переходы с новыми состояниями.
 * @param start_temp_index начальный индекс для новых состояний.
 * @return Пара: вектор обратных переходов и вектор новых состояний.
 */
std::pair<std::vector<PDAState::Transitions>, std::vector<PDAState>> 
PushdownAutomaton::get_reversed_transitions_with_new_states(int start_temp_index) const {
	// Результат будет содержать больше состояний, чем исходный автомат
	vector<PDAState::Transitions> res(size());
	vector<PDAState> new_temp_states; // новые промежуточные состояния
	int next_temp_state_index = (start_temp_index == -1) ? size() : start_temp_index;

	// проходимся по всему изначальному автомату
	for (int i = 0; i < size(); ++i) {
		// для каждого состояния рассматриваем переходы на каждый символ
		for (const auto& [symbol, symbol_transitions] : states[i].transitions) {
			for (const auto& tr : symbol_transitions) {
				// Обратный переход: меняем направление и операции со стеком
				
				// Если мы на стек клали просто eps, то pop = epsilon, push = {исходный pop}
				// без каких-либо дополнительных состояний
				if (tr.push.empty()) {
					PDATransition reversed_tr(i, tr.input_symbol, Symbol::Epsilon, {tr.pop});
					if (static_cast<size_t>(tr.to) >= res.size()) res.resize(tr.to + 1);
					res[tr.to][symbol].insert(reversed_tr);

				// если же push содержит один символ, то просто меняем pop и push местами
				} else if (tr.push.size() == 1) {
					PDATransition reversed_tr(i, tr.input_symbol, tr.push[0], {tr.pop});
					if (static_cast<size_t>(tr.to) >= res.size()) res.resize(tr.to + 1);
					res[tr.to][symbol].insert(reversed_tr);
				
				// Если push содержит несколько символов [z0, A, B], то на стеке они будут B, A, z0 (сверху вниз)
				// В reverse нужно снимать в обратном порядке: B, A, z0
				// Поэтому итерируемся по push В ОБРАТНОМ ПОРЯДКЕ: push[last], push[last-1], ..., push[0]
				// Создаём цепочку: tr.to --symbol, pop=push[last], push=ε--> temp0 --ε, pop=push[last-1], push=ε--> ... --ε, pop=push[0], push={tr.pop}--> i
				} else {
					// Индекс первого временного состояния
					int first_temp_index = next_temp_state_index;
					
					// Создаём переход из tr.to в первое временное состояние (снимаем последний элемент push)
					PDATransition first_tr(first_temp_index, tr.input_symbol, tr.push[tr.push.size()-1], {});
					// Расширяем при необходимости вектор 
					if (static_cast<size_t>(tr.to) >= res.size()) res.resize(tr.to + 1);
					// Добавляем этот переход в результат
					res[tr.to][symbol].insert(first_tr);
					
					// Создаём промежуточные состояния и epsilon-переходы (итерируемся в обратном порядке)
					for (int j = tr.push.size() - 2; j >= 0; --j) {
						int current_temp_index = next_temp_state_index++;
						std::ostringstream oss;
						oss << "tmp" << i << "s" << tr.to << "p" << (tr.push.size() - 1 - j);
						
						// Создаём новое промежуточное состояние
						PDAState temp_state(current_temp_index, oss.str(), false);
						
						if (j > 0) {
							// Промежуточный переход: pop следующий символ из push (в обратном порядке)
							int next_temp = next_temp_state_index;
							PDATransition eps_tr(next_temp, Symbol::Epsilon, tr.push[j], {});
							temp_state.transitions[Symbol::Epsilon].insert(eps_tr);
						} else {
							// Последний переход: pop push[0], push = {исходный pop}, идём в i
							// идем в i - потому что мы смотрели от лица i->to - чтобы правильно
							// развернуть переходы все
							PDATransition last_tr(i, Symbol::Epsilon, tr.push[j], {tr.pop});
							temp_state.transitions[Symbol::Epsilon].insert(last_tr);
						}
						
						// добавляем созданное временное состояние
						new_temp_states.push_back(temp_state);
					}
				}
			}
		}
	}

	return {res, new_temp_states};
}

PushdownAutomaton PushdownAutomaton::reverse(iLogTemplate* log) const {
	// создаем новый PDA с теми же состояниями и алфавитом
	PushdownAutomaton new_pda(size(), states, language->get_alphabet());
	// считаем количество финальных состояний
	int final_states_counter = count_if(new_pda.states.begin(),
										new_pda.states.end(),
										[](const PDAState& state) { return state.is_terminal; });

	// Если финальных состояний > 1, то добавляем новое состояние RevS
	if (final_states_counter > 1) {
		new_pda.states.push_back({(int)new_pda.size(),
								  "RevS",
								  false,
								  PDAState::Transitions()});
	}

	// Если финальных состояний > 0, то начинаем reversing
	if (final_states_counter > 0) {
		
		// если > 1 финального состояния, значит было добавлено новое стартовое RevS
		int final_states_flag = (final_states_counter > 1) ? 1 : 0;
		int new_initial_idx = (final_states_counter > 1) ? (new_pda.size() - 1) : -1;

		// не трогаем последнее (RevS), если мы его добавили
		for (int i = 0; i < new_pda.size() - final_states_flag; ++i) {
			if (new_pda.states[i].is_terminal) {
				// ранее финальное теряет свое свойство
				new_pda.states[i].is_terminal = false;
				if (final_states_counter == 1) {
					new_pda.initial_state = i;
					new_initial_idx = i;
				}
			}
		}
		// начальное состояние становится финальным
		new_pda.states[initial_state].is_terminal = true;
		
		// если же у нас было только 1 финальное, тогда делаем его начальным
		if (final_states_counter > 1) {
			new_pda.initial_state = new_initial_idx;
		}

		// Получаем обратные переходы и новые промежуточные состояния
		int start_temp_index = new_pda.size();
		auto [new_transition_matrix, temp_states] = get_reversed_transitions_with_new_states(start_temp_index);
		
		// Применяем обратные переходы к существующим состояниям
		for (int i = 0; i < new_pda.size() - final_states_flag; ++i) {
			new_pda.states[i].transitions = new_transition_matrix[i];
		}
		
		// Добавляем новые промежуточные состояния
		for (auto& temp_state : temp_states) {
			new_pda.states.push_back(temp_state);
		}

		// Добавляем переходы из RevS (или единственного старого финального состояния) 
		// для генерации любого содержимого стека (так как мы принимаем не обязательно по пустому стеку)
		auto stack_symbols = _get_stack_symbols();
		for (const auto& sym : stack_symbols) {
			// RevS -> RevS, read eps, pop eps, push sym
			PDATransition loop_tr(new_initial_idx, Symbol::Epsilon, Symbol::Epsilon, {sym});
			new_pda.states[new_initial_idx].transitions[Symbol::Epsilon].insert(loop_tr);
		}

		// Если мы создали RevS, то соединяем его со старыми финальными состояниями
		if (final_states_counter > 1) {
			for (int i = 0; i < size(); ++i) {
				if (states[i].is_terminal) {
					// RevS -> old final (eps, eps -> eps)
					PDATransition eps_tr(i, Symbol::Epsilon, Symbol::Epsilon, {Symbol::Epsilon});
					new_pda.states[new_initial_idx].transitions[Symbol::Epsilon].insert(eps_tr);
				}
			}
		}
	} else {
		new_pda.initial_state = initial_state;
	}

	// Удаляем недостижимые состояния
	new_pda = new_pda._remove_unreachable_states(log);
    std::cerr << "Reversed PDA:\n" << new_pda.to_txt() << std::endl;
	if (log) {
		log->set_parameter("oldautomaton", *this);
		log->set_parameter("result", new_pda);
	}
	return new_pda;
}
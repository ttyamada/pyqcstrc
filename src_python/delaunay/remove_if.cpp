namespace std {
  template <class ForwardIterator, class Predicate>
  ForwardIterator
    remove_if(ForwardIterator first,
              ForwardIterator last,
              Predicate pred);        // (1) C++03

  template <class ForwardIterator, class Predicate>
  constexpr ForwardIterator
    remove_if(ForwardIterator first,
              ForwardIterator last,
              Predicate pred);        // (1) C++20

  template <class ExecutionPolicy, class ForwardIterator, class Predicate>
  ForwardIterator
    remove_if(ExecutionPolicy&& exec,
              ForwardIterator first,
              ForwardIterator last,
              Predicate pred);        // (2) C++17
}

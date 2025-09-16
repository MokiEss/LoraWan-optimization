set(base_examples
    run_simulation
)

foreach(example ${base_examples})
  add_executable(${example} ${example}.cc)

  # Link against lorawan + ns3 libraries
  target_link_libraries(${example}
    PRIVATE
      ${liblorawan}
      ${libcore}
      ${libnetwork}
      ${libmobility}
      ${libinternet}
  )
endforeach()

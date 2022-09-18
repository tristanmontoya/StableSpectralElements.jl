function CLOUD_reset_timer()
    for t in 1:Threads.nthreads()
        thread_timer = get_timer(string("thread_timer_",t))
        reset_timer!(thread_timer)
    end
    to = merge(Tuple(get_timer(string("thread_timer_",t)) 
    for t in 1:Threads.nthreads())...)
    reset_timer!(to)
end

function CLOUD_print_timer()
    to = merge(Tuple(get_timer(string("thread_timer_",t)) 
    for t in 1:Threads.nthreads())...)
    print_timer(to)
end
    